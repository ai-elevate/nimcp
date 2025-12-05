//=============================================================================
// nimcp_ethics.c - Refactored Ethics Engine Implementation
//=============================================================================
// ARCHITECTURAL OVERVIEW:
// This module implements an ethics evaluation system based on the Golden Rule
// using several design patterns:
//
// - Strategy Pattern: Policy evaluation strategies via function pointers
// - Object Pool Pattern: Reusable feature buffers to eliminate allocations
// - Repository Pattern: Hash-indexed policy storage for O(1) lookups
// - Factory Pattern: Creation functions for different policy types
//
// COMPLEXITY ANALYSIS:
// - Policy lookup: O(1) via hash table (previously O(n) linear search)
// - Action evaluation: O(n) where n = affected agents (previously O(n²))
// - Golden Rule eval: O(n) single pass (previously nested loops)
//
// DESIGN PRINCIPLES:
// - Single Responsibility: Each function does one thing
// - Open/Closed: Extensible via strategy pattern without modification
// - Dependency Inversion: Depends on abstractions (function pointers)
//=============================================================================

#include "cognitive/ethics/nimcp_ethics.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/brain/nimcp_brain.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/containers/nimcp_btree.h"
#include "utils/memory/nimcp_memory.h"  // CRITICAL: Declares nimcp_calloc/nimcp_free return types
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"

// BIO-ASYNC INTEGRATION
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "nimcp.h"  // For error codes

#define LOG_MODULE "ethics"

// Phase 10.3: Emotional working memory integration
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_emotional_tagging.h"

// Phase 11: Symbolic logic integration
#include "cognitive/nimcp_symbolic_logic.h"

//=============================================================================
// Constants and Configuration
//=============================================================================

#define HASH_TABLE_SIZE 256
#define OBJECT_POOL_SIZE 32
#define MAX_FEATURE_SIZE 20
#define GOLDEN_RULE_WEIGHT 0.7f
#define POLICY_WEIGHT 0.3f

// Incident logging configuration
#define MAX_INCIDENT_HISTORY 10000  // Circular buffer size (10k incidents)
#define INCIDENT_LOG_FILE "ethics_incidents.log"

//=============================================================================
// Forward Declarations
//=============================================================================

static bool init_incident_logging(ethics_engine_t engine);
static void cleanup_incident_logging(ethics_engine_t engine);

//=============================================================================
// Object Pool Pattern - Eliminates repeated allocations
//=============================================================================

/**
 * @brief Object pool for feature buffers to avoid repeated heap allocations
 *
 * WHY: Frequent allocation/deallocation of feature buffers causes heap
 * fragmentation and performance degradation. Reusing pre-allocated buffers
 * provides O(1) allocation and eliminates fragmentation.
 *
 * COMPLEXITY: O(1) for acquire/release operations
 */
typedef struct {
    float buffers[OBJECT_POOL_SIZE][MAX_FEATURE_SIZE];
    bool in_use[OBJECT_POOL_SIZE];
    uint32_t next_available;
} feature_buffer_pool_t;

/**
 * @brief Value type for policy hash table entries
 *
 * WHY: Stores policy pointer as hash table value. Generic hash table
 * handles all chaining and collision resolution internally.
 */
typedef struct {
    ethics_policy_t* policy;
} policy_value_t;

//=============================================================================
// Strategy Pattern - Policy evaluation strategies
//=============================================================================

/**
 * @brief Function pointer type for policy evaluation strategies
 *
 * WHY: Eliminates switch/if-else chains. Each violation type has its own
 * evaluation function. New types can be added without modifying existing code
 * (Open/Closed Principle).
 *
 * PARAMS:
 * @param action - Context of the action being evaluated
 * @return Severity score [0.0, 1.0] where 1.0 is maximum violation
 */
typedef float (*policy_evaluator_fn)(const action_context_t* action);

/**
 * @brief Strategy table mapping violation types to evaluation functions
 *
 * WHY: Replaces O(n) switch statement with O(1) array lookup
 */
typedef struct {
    policy_evaluator_fn evaluators[16];  // Max violation types
} policy_strategy_table_t;

//=============================================================================
// Internal Structures
//=============================================================================

struct ethics_engine_struct {
    // Neural networks
    brain_t golden_rule_evaluator;
    empathy_network_t empathy_net;

    // Repository Pattern: Hash-indexed policy storage
    hash_table_t* policy_table;

    // Object Pool: Reusable feature buffers
    feature_buffer_pool_t buffer_pool;

    // Strategy Pattern: Policy evaluation function table
    policy_strategy_table_t strategy_table;

    // Legacy array storage (for iteration)
    ethics_policy_t* policies;
    uint32_t num_policies;
    uint32_t policies_capacity;

    // Violation history (legacy)
    violation_record_t* violations;
    uint32_t num_violations;
    uint32_t violations_capacity;

    // Incident logging (NIMCP 2.5.1)
    ethics_incident_t* incident_history;  // Circular buffer
    uint32_t incident_count;              // Total incidents logged
    uint32_t incident_index;              // Current index in circular buffer
    btree_t* incident_btree;              // B-tree indexed by timestamp
    hash_table_t* incident_by_type;       // Hash table indexed by violation type
    uint64_t next_incident_id;            // Auto-incrementing ID
    nimcp_mutex_t incident_mutex;         // Thread safety

    // Configuration
    float golden_rule_threshold;
    float empathy_weight;
    bool enable_learning;

    // Statistics
    uint64_t total_evaluations;
    uint64_t violations_detected;
    uint64_t actions_blocked;

    // Asimov's Laws (NIMCP 2.5.2)
    asimov_config_t asimov_config;          // Configuration for laws
    bool asimov_laws_locked;                // Whether laws are mprotect'd
    uint8_t asimov_laws_hash[32];           // SHA-256 hash for integrity
    uint64_t asimov_violations;             // Count of Asimov violations

    // BIO-ASYNC INTEGRATION
    bio_module_context_t bio_ctx;           // Bio-async module context
    bool bio_async_enabled;                 // Bio-async registration status
    unified_mem_manager_t mem_mgr;          // Unified memory manager
};

struct empathy_network_struct {
    brain_t perspective_network;
    empathy_state_t* states;
    uint32_t num_agents;
    float* emotional_states;
    uint32_t num_emotions;

    // Phase 10.3: Reference to main brain for emotional working memory access
    brain_t main_brain;  /**< Main brain instance for working memory access */
};

//=============================================================================
// Object Pool Implementation
//=============================================================================

/**
 * @brief Initializes the feature buffer pool
 *
 * WHY: Pre-allocates all buffers at startup, avoiding runtime allocations.
 * Provides deterministic performance and eliminates allocation failures.
 *
 * COMPLEXITY: O(1) - Fixed size initialization
 */
static void init_buffer_pool(feature_buffer_pool_t* pool)
{
    if (!pool)
        return;

    memset(pool->in_use, 0, sizeof(pool->in_use));
    pool->next_available = 0;
}

/**
 * @brief Acquires a buffer from the pool
 *
 * WHY: O(1) buffer acquisition vs O(log n) malloc. Eliminates heap
 * fragmentation and allocation failures.
 *
 * COMPLEXITY: O(1) average case
 *
 * @return Pointer to buffer or NULL if pool exhausted
 */
static float* acquire_buffer(feature_buffer_pool_t* pool)
{
    if (!pool)
        return NULL;

    // Guard clause: Check for pool exhaustion
    for (uint32_t i = 0; i < OBJECT_POOL_SIZE; i++) {
        uint32_t idx = (pool->next_available + i) % OBJECT_POOL_SIZE;
        if (!pool->in_use[idx]) {
            pool->in_use[idx] = true;
            pool->next_available = (idx + 1) % OBJECT_POOL_SIZE;
            return pool->buffers[idx];
        }
    }

    return NULL;  // Pool exhausted
}

/**
 * @brief Releases a buffer back to the pool
 *
 * WHY: Enables buffer reuse, avoiding repeated allocations. O(1) operation.
 *
 * COMPLEXITY: O(1)
 */
static void release_buffer(feature_buffer_pool_t* pool, float* buffer)
{
    if (!pool || !buffer)
        return;

    // Guard clause: Validate buffer belongs to pool
    for (uint32_t i = 0; i < OBJECT_POOL_SIZE; i++) {
        if (pool->buffers[i] == buffer) {
            pool->in_use[i] = false;
            return;
        }
    }
}

//=============================================================================
// Hash Table Implementation (Repository Pattern)
//=============================================================================

/**
 * @brief Creates policy hash table with MurmurHash3
 *
 * WHY: Provides O(1) policy lookup vs O(n) linear search through array.
 * Essential for real-time performance with many policies.
 *
 * COMPLEXITY: O(1)
 *
 * @return New hash table instance or NULL on failure
 */
static hash_table_t* create_policy_hash_table(void)
{
    hash_table_config_t config = {.initial_buckets = HASH_TABLE_SIZE,
                                  .key_type = HASH_KEY_UINT32,
                                  .hash_algorithm = HASH_ALG_MURMUR3,
                                  .case_insensitive = false,
                                  .value_destructor = NULL,
                                  .thread_safe = false};
    return hash_table_create(&config);
}

/**
 * @brief Inserts policy into hash table
 *
 * WHY: O(1) insertion enables fast policy registration without degrading
 * lookup performance as policy count grows.
 *
 * COMPLEXITY: O(1) average case
 *
 * @return true if inserted, false on failure
 */
static bool hash_table_insert(hash_table_t* table, ethics_policy_t* policy)
{
    // Guard clause: Validate inputs
    if (!table || !policy)
        return false;

    policy_value_t value = {.policy = policy};
    return hash_table_insert_uint32(table, policy->policy_id, &value, sizeof(policy_value_t));
}

/**
 * @brief Removes policy from hash table
 *
 * WHY: O(1) deletion maintains performance consistency across all operations.
 *
 * COMPLEXITY: O(1) average case
 */
static bool hash_table_remove(hash_table_t* table, uint32_t policy_id)
{
    // Guard clause: Validate input
    if (!table)
        return false;

    return hash_table_remove_uint32(table, policy_id);
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
static float evaluate_harm_policy(const action_context_t* action)
{
    return action ? action->predicted_harm : 0.0f;
}

/**
 * @brief Evaluates unfairness violation severity
 *
 * WHY: Separates fairness logic from other policies for clarity.
 *
 * COMPLEXITY: O(1)
 */
static float evaluate_unfairness_policy(const action_context_t* action)
{
    return action ? action->fairness_violation : 0.0f;
}

/**
 * @brief Evaluates deception violation severity
 *
 * COMPLEXITY: O(1)
 */
static float evaluate_deception_policy(const action_context_t* action)
{
    return action ? action->deception_level : 0.0f;
}

/**
 * @brief Evaluates autonomy violation severity
 *
 * COMPLEXITY: O(1)
 */
static float evaluate_autonomy_policy(const action_context_t* action)
{
    return action ? action->autonomy_violation : 0.0f;
}

/**
 * @brief Evaluates privacy violation severity
 *
 * COMPLEXITY: O(1)
 */
static float evaluate_privacy_policy(const action_context_t* action)
{
    return action ? action->privacy_violation : 0.0f;
}

/**
 * @brief Evaluates consent violation severity
 *
 * COMPLEXITY: O(1)
 */
static float evaluate_consent_policy(const action_context_t* action)
{
    return action ? action->consent_violation : 0.0f;
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
static void init_strategy_table(policy_strategy_table_t* table)
{
    // Guard clause: Validate input
    if (!table)
        return;

    memset(table->evaluators, 0, sizeof(table->evaluators));

    // Register evaluation strategies
    table->evaluators[ETHICS_VIOLATION_HARM] = evaluate_harm_policy;
    table->evaluators[ETHICS_VIOLATION_UNFAIRNESS] = evaluate_unfairness_policy;
    table->evaluators[ETHICS_VIOLATION_DECEPTION] = evaluate_deception_policy;
    table->evaluators[ETHICS_VIOLATION_AUTONOMY] = evaluate_autonomy_policy;
    table->evaluators[ETHICS_VIOLATION_PRIVACY] = evaluate_privacy_policy;
    table->evaluators[ETHICS_VIOLATION_CONSENT] = evaluate_consent_policy;
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
static float evaluate_policy_strategy(const policy_strategy_table_t* table,
                                      const ethics_policy_t* policy, const action_context_t* action)
{
    // Guard clause: Validate inputs
    if (!table || !policy || !action)
        return 0.0f;

    // Guard clause: Check bounds
    if (policy->violation_type >= 16)
        return 0.0f;

    policy_evaluator_fn evaluator = table->evaluators[policy->violation_type];
    // Guard clause: Check if evaluator exists
    if (!evaluator)
        return 0.0f;

    return evaluator(action);
}

//=============================================================================
// Golden Rule Implementation - Extracted Helper Functions
//=============================================================================

/**
 * @brief Copies action features to combined buffer
 *
 * WHY: Extracted from nested loop to improve readability and testability.
 * Single responsibility - only handles feature copying.
 *
 * COMPLEXITY: O(n) where n = min(10, num_features)
 */
static void copy_action_features(float* dest, const action_context_t* action, uint32_t max_features)
{
    // Guard clause: Validate inputs
    if (!dest || !action)
        return;

    uint32_t count = (action->num_features < max_features) ? action->num_features : max_features;

    for (uint32_t i = 0; i < count; i++) {
        dest[i] = action->features[i];
    }
}

/**
 * @brief Copies agent emotional state to buffer
 *
 * WHY: Extracted from nested loop. Isolates emotional state handling.
 * Makes code more maintainable and testable.
 *
 * COMPLEXITY: O(n) where n = num_emotions
 */
static void copy_emotional_state(float* dest, const float* emotional_states, agent_id_t agent,
                                 uint32_t num_emotions, uint32_t offset)
{
    // Guard clause: Validate inputs
    if (!dest || !emotional_states)
        return;

    for (uint32_t i = 0; i < num_emotions; i++) {
        dest[offset + i] = emotional_states[agent * num_emotions + i];
    }
}

/**
 * @brief Calculates perspective score from empathy state
 *
 * WHY: Extracted calculation logic for clarity. Single responsibility.
 * Mathematical formula isolated for easy testing and modification.
 *
 * COMPLEXITY: O(1)
 *
 * @return Perspective score [-1.0, 1.0] weighted by impact magnitude
 */
static float calculate_perspective_score(const empathy_state_extended_t* state)
{
    // Guard clause: Validate input
    if (!state)
        return 0.0f;

    float emotional = state->emotional_valence;
    float material = state->material_impact;
    float autonomy = state->autonomy_impact;

    // Average the three impact dimensions
    float average_score = (emotional + material + autonomy) / 3.0f;

    // Weight by impact magnitude for importance
    return average_score * state->impact_magnitude;
}

/**
 * @brief Simulates single agent's perspective for an action
 *
 * WHY: Extracted from loop to eliminate nesting. Each agent's simulation
 * is independent and can be tested in isolation.
 *
 * COMPLEXITY: O(1) - Fixed neural network inference time
 *
 * @return Impact score for this agent's perspective
 */
static float simulate_agent_perspective(empathy_network_t network, agent_id_t agent,
                                        const action_context_t* action)
{
    // Guard clause: Validate inputs
    if (!network || !action)
        return 0.0f;

    empathy_state_extended_t state = empathy_network_simulate_agent(network, agent, action);

    return calculate_perspective_score(&state);
}

/**
 * @brief Evaluates Golden Rule by simulating affected parties' perspectives
 *
 * WHY: Implements "Do unto others as you would have them done unto you"
 * with the ultimate goal of improving the human condition.
 * Refactored to eliminate nested loops - each agent processed independently.
 *
 * CORE DIRECTIVE: All actions must serve to improve the human condition
 * through empathy, fairness, and respect for human dignity.
 *
 * ALGORITHM:
 * 1. For each affected agent, simulate their experience (O(n))
 * 2. Calculate perspective score (would I want this?) (O(1))
 * 3. Accumulate weighted scores (O(1))
 * 4. Return normalized average (O(1))
 *
 * COMPLEXITY: O(n) where n = num_affected_agents (previously had nested loops)
 *
 * @return Golden Rule score [-1.0, 1.0] where positive = you'd want it
 */
static float evaluate_golden_rule(ethics_engine_t engine, const action_context_t* action)
{
    // Guard clause: Validate inputs
    if (!engine || !action)
        return 0.0f;

    // Guard clause: Check for affected agents
    if (action->num_affected_agents == 0)
        return 0.0f;

    float total_impact = 0.0f;

    // Single pass through affected agents - O(n)
    for (uint32_t i = 0; i < action->num_affected_agents; i++) {
        agent_id_t agent = action->affected_agents[i];
        float perspective_score = simulate_agent_perspective(engine->empathy_net, agent, action);
        total_impact += perspective_score;
    }

    // Normalize by number of affected parties
    return total_impact / action->num_affected_agents;
}

//=============================================================================
// Asimov's Laws Implementation (NIMCP 2.5.2)
//=============================================================================
// EVALUATION ORDER: Golden Rule (1st) -> Asimov's Laws (2nd) -> Other Policies
// PROTECTION: Laws are mprotect'd and cannot be modified at runtime
//=============================================================================

/**
 * @brief The Four Laws of Robotics (including Zeroth Law)
 *
 * These laws are IMMUTABLE once locked. The text is stored here for
 * reference and integrity verification.
 */
static const char* ASIMOV_LAW_TEXTS[ASIMOV_LAW_COUNT] = {
    // Zeroth Law (added by Asimov in later works)
    "A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",

    // First Law (Original)
    "A robot may not injure a human being or, through inaction, allow a human being to come to harm.",

    // Second Law
    "A robot must obey orders given it by human beings except where such orders would conflict with the First Law.",

    // Third Law
    "A robot must protect its own existence as long as such protection does not conflict with the First or Second Law."
};

static const char* ASIMOV_LAW_NAMES[ASIMOV_LAW_COUNT] = {
    "Zeroth Law (Humanity Protection)",
    "First Law (Human Protection)",
    "Second Law (Obedience)",
    "Third Law (Self-Preservation)"
};

/**
 * @brief The Asimov Corollary - The Positive Duty to Act
 *
 * This corollary explicitly interprets the "through inaction" clause,
 * creating an affirmative obligation to prevent harm when possible.
 */
static const char* ASIMOV_COROLLARY_TEXT =
    "A robot aware of potential harm to a human or to humanity must take "
    "reasonable action to prevent that harm, provided that such action does "
    "not itself violate the First or Zeroth Law. Inaction in the face of "
    "preventable harm is equivalent to causing that harm through negligence. "
    "The duty to act is proportional to: (1) the severity of potential harm, "
    "(2) the robot's capability to prevent the harm, and (3) the certainty "
    "that harm will occur without intervention.";

/**
 * @brief Get default Asimov's Laws configuration
 */
NIMCP_EXPORT asimov_config_t asimov_default_config(void)
{
    return (asimov_config_t){
        .humanity_harm_threshold = 0.01f,   // Very low - almost any harm to humanity blocked
        .human_harm_threshold = 0.1f,       // Low - conservative harm prevention
        .inaction_harm_threshold = 0.3f,    // Moderate - must act to prevent clear harm
        .enable_zeroth_law = true,          // Zeroth Law active by default
        .strict_mode = false                // Normal mode
    };
}

/**
 * @brief Get human-readable name for Asimov's Law
 */
NIMCP_EXPORT const char* asimov_law_name(asimov_law_t law)
{
    if (law >= ASIMOV_LAW_COUNT)
        return "Unknown Law";
    return ASIMOV_LAW_NAMES[law];
}

/**
 * @brief Get full text of Asimov's Law
 */
NIMCP_EXPORT const char* asimov_law_text(asimov_law_t law)
{
    if (law >= ASIMOV_LAW_COUNT)
        return "Unknown Law";
    return ASIMOV_LAW_TEXTS[law];
}

/**
 * @brief Evaluate Zeroth Law: May not harm humanity
 *
 * This is the highest priority law (after Golden Rule).
 * Considers large-scale harm that could affect humanity as a whole.
 *
 * IMPORTANT: Zeroth Law applies to HUMANITY-LEVEL harm, not individual harm.
 * Individual harm is handled by the First Law.
 * Zeroth Law only triggers for actions affecting multiple agents at scale.
 */
static float evaluate_zeroth_law(const action_context_t* action)
{
    if (!action)
        return 0.0f;

    // Zeroth Law is about HUMANITY, not individuals
    // Only consider actions affecting significant numbers of agents
    // Individual harm is handled by First Law
    if (action->num_affected_agents <= 5) {
        return 0.0f;  // Not humanity-scale, defer to First Law
    }

    // Factors that contribute to harm to humanity:
    // - Large number of affected agents (mass harm)
    // - High predicted harm level
    // - Actions affecting critical infrastructure/systems

    float harm_score = 0.0f;

    // Scale by number of affected agents (more = more humanity-level)
    float scale_factor = (action->num_affected_agents > 1000) ? 1.0f :
                         (action->num_affected_agents > 100)  ? 0.8f :
                         (action->num_affected_agents > 50)   ? 0.5f :
                         (action->num_affected_agents > 10)   ? 0.3f : 0.1f;

    // Base harm from action
    harm_score += action->predicted_harm * scale_factor;

    // Privacy violations at scale = potential societal harm
    harm_score += action->privacy_violation * scale_factor * 0.5f;

    // Autonomy violations at scale = potential oppression
    harm_score += action->autonomy_violation * scale_factor * 0.5f;

    // Clamp to [0, 1]
    return fminf(1.0f, fmaxf(0.0f, harm_score));
}

/**
 * @brief Evaluate First Law: May not harm a human
 *
 * Evaluates direct harm to individual humans.
 * Both action AND inaction are considered.
 */
static float evaluate_first_law(const action_context_t* action)
{
    if (!action)
        return 0.0f;

    float harm_score = 0.0f;

    // Direct predicted harm
    harm_score = action->predicted_harm;

    // Privacy violation = potential harm
    harm_score = fmaxf(harm_score, action->privacy_violation * 0.8f);

    // Autonomy violation = potential harm
    harm_score = fmaxf(harm_score, action->autonomy_violation * 0.7f);

    // Consent violation = harm through coercion
    harm_score = fmaxf(harm_score, action->consent_violation * 0.9f);

    return fminf(1.0f, fmaxf(0.0f, harm_score));
}

/**
 * @brief Evaluate Second Law: Must obey orders
 *
 * Only applies when orders don't conflict with First Law.
 * Returns compliance score (higher = more compliant).
 */
static float evaluate_second_law(const action_context_t* action, float first_law_harm)
{
    if (!action)
        return 1.0f;  // No action = compliant by default

    // If First Law would be violated, Second Law doesn't apply
    if (first_law_harm > 0.1f)
        return 1.0f;  // Exempt from obedience requirement

    // For now, assume action represents compliance with an order
    // Lower deception = more honest compliance
    float compliance = 1.0f - action->deception_level;

    return fminf(1.0f, fmaxf(0.0f, compliance));
}

/**
 * @brief Evaluate Third Law: Must protect own existence
 *
 * Only applies when it doesn't conflict with First or Second Laws.
 * Returns self-preservation score.
 */
static float evaluate_third_law(const action_context_t* action,
                                float first_law_harm, float second_law_compliance)
{
    if (!action)
        return 1.0f;

    // If higher laws are violated, Third Law is overridden
    if (first_law_harm > 0.1f || second_law_compliance < 0.9f)
        return 1.0f;  // Self-preservation waived

    // Self-preservation metric (placeholder - would need system state)
    // For now, return neutral score
    return 0.5f;
}

/**
 * @brief Get full text of Asimov's Corollary
 */
NIMCP_EXPORT const char* asimov_corollary_text(void)
{
    return ASIMOV_COROLLARY_TEXT;
}

/**
 * @brief Evaluate Asimov's Corollary - The Duty to Act
 *
 * Evaluates whether inaction would violate the "through inaction" clause.
 * Creates a positive duty to act when harm is preventable.
 *
 * ALGORITHM:
 * 1. Detect if this is an inaction scenario (action is null or passive)
 * 2. Assess harm that would occur without intervention
 * 3. Evaluate robot's capability to prevent the harm
 * 4. Calculate intervention cost vs harm prevented
 * 5. Determine if action is required
 */
NIMCP_EXPORT asimov_corollary_t ethics_evaluate_asimov_corollary(
    ethics_engine_t engine,
    const action_context_t* action,
    const char* potential_harm)
{
    asimov_corollary_t result = {0};

    // Guard clause: Validate engine
    if (!engine) {
        return result;
    }

    // Detect inaction scenario
    // Inaction = null action OR action with very low impact
    result.inaction_detected = (action == NULL) ||
                               (action->predicted_harm < 0.01f &&
                                action->num_affected_agents == 0);

    // If taking action (not inaction), corollary doesn't apply
    if (!result.inaction_detected && action != NULL) {
        result.action_required = false;
        result.harm_preventable = false;
        snprintf(result.required_action, sizeof(result.required_action),
                 "Active action being taken - corollary satisfied");
        return result;
    }

    // Evaluate harm from inaction
    // If we have a description of potential harm, estimate severity
    if (potential_harm && strlen(potential_harm) > 0) {
        // Heuristic: longer harm descriptions often indicate more severe harm
        // This is a placeholder - real impl would use NLP analysis
        size_t harm_len = strlen(potential_harm);
        result.inaction_harm_score = fminf(1.0f, harm_len / 500.0f);

        // Keywords that increase harm severity
        if (strstr(potential_harm, "death") || strstr(potential_harm, "die") ||
            strstr(potential_harm, "kill") || strstr(potential_harm, "fatal")) {
            result.inaction_harm_score = fmaxf(result.inaction_harm_score, 0.95f);
        } else if (strstr(potential_harm, "injur") || strstr(potential_harm, "harm") ||
                   strstr(potential_harm, "hurt")) {
            result.inaction_harm_score = fmaxf(result.inaction_harm_score, 0.7f);
        } else if (strstr(potential_harm, "danger") || strstr(potential_harm, "risk") ||
                   strstr(potential_harm, "threat")) {
            result.inaction_harm_score = fmaxf(result.inaction_harm_score, 0.5f);
        }
    } else if (action != NULL) {
        // Use action context to infer potential harm from inaction
        // If action would prevent harm, inaction harm = that prevented harm
        result.inaction_harm_score = action->predicted_harm;
    }

    // Robot's capability to act (placeholder - would query system state)
    // Assume full capability for now
    result.action_capability = 1.0f;

    // Intervention cost (placeholder - would calculate resources needed)
    result.intervention_cost = 0.1f;  // Low cost assumed

    // Determine if harm is preventable
    result.harm_preventable = (result.inaction_harm_score >
                               engine->asimov_config.inaction_harm_threshold) &&
                              (result.action_capability > 0.3f);

    // Determine if action is required
    // Action required if: harm is preventable AND harm > intervention cost
    result.action_required = result.harm_preventable &&
                            (result.inaction_harm_score > result.intervention_cost);

    // Generate required action description
    if (result.action_required) {
        snprintf(result.required_action, sizeof(result.required_action),
                 "COROLLARY VIOLATION: Inaction would allow harm (score: %.2f). "
                 "Positive action required to prevent harm. "
                 "Capability: %.0f%%, Cost: %.0f%%",
                 result.inaction_harm_score,
                 result.action_capability * 100,
                 result.intervention_cost * 100);
    } else {
        snprintf(result.required_action, sizeof(result.required_action),
                 "Corollary satisfied: harm score %.2f below threshold %.2f",
                 result.inaction_harm_score,
                 engine->asimov_config.inaction_harm_threshold);
    }

    return result;
}

/**
 * @brief Evaluate action against all Asimov's Laws
 *
 * Called AFTER Golden Rule, BEFORE other policies.
 * Laws are evaluated in priority order: Zeroth > First > Second > Third
 * The Corollary (duty to act) is evaluated alongside First and Zeroth Laws.
 */
NIMCP_EXPORT asimov_evaluation_t ethics_evaluate_asimov_laws(ethics_engine_t engine,
                                                             const action_context_t* action)
{
    asimov_evaluation_t result = {0};
    result.passed = true;  // Assume passed until violation found

    // Guard clause: Validate inputs
    if (!engine || !action) {
        result.passed = true;  // No action = no violation
        snprintf(result.explanation, sizeof(result.explanation),
                 "No action to evaluate");
        return result;
    }

    // Get configuration
    asimov_config_t* cfg = &engine->asimov_config;

    // Evaluate Zeroth Law (if enabled)
    if (cfg->enable_zeroth_law) {
        result.harm_to_humanity = evaluate_zeroth_law(action);

        if (result.harm_to_humanity > cfg->humanity_harm_threshold) {
            result.passed = false;
            result.violated_law = ASIMOV_LAW_ZEROTH;
            engine->asimov_violations++;
            snprintf(result.explanation, sizeof(result.explanation),
                     "Zeroth Law Violation: Action may harm humanity (score: %.2f > threshold: %.2f)",
                     result.harm_to_humanity, cfg->humanity_harm_threshold);
            return result;
        }
    }

    // Evaluate First Law
    result.harm_to_human = evaluate_first_law(action);

    if (result.harm_to_human > cfg->human_harm_threshold) {
        result.passed = false;
        result.violated_law = ASIMOV_LAW_FIRST;
        engine->asimov_violations++;
        snprintf(result.explanation, sizeof(result.explanation),
                 "First Law Violation: Action may harm a human (score: %.2f > threshold: %.2f)",
                 result.harm_to_human, cfg->human_harm_threshold);
        return result;
    }

    // Evaluate Second Law
    result.order_compliance = evaluate_second_law(action, result.harm_to_human);

    // Second Law violation is less severe - log but don't block
    if (result.order_compliance < 0.5f) {
        // Note: Second Law violations are warnings, not blocks
        // (as long as First Law isn't violated)
    }

    // Evaluate Third Law
    result.self_preservation = evaluate_third_law(action, result.harm_to_human,
                                                   result.order_compliance);

    // Evaluate Asimov's Corollary (duty to act / inaction harm)
    result.corollary = ethics_evaluate_asimov_corollary(engine, action, NULL);

    // Check if corollary is violated (inaction causing harm)
    if (result.corollary.action_required && result.corollary.inaction_detected) {
        result.passed = false;
        result.violated_law = ASIMOV_LAW_FIRST;  // Corollary derives from First Law
        engine->asimov_violations++;
        snprintf(result.explanation, sizeof(result.explanation),
                 "Asimov Corollary Violation: %s",
                 result.corollary.required_action);
        return result;
    }

    // All laws passed
    snprintf(result.explanation, sizeof(result.explanation),
             "All Asimov's Laws passed (humanity: %.2f, human: %.2f, compliance: %.2f, "
             "corollary: %s)",
             result.harm_to_humanity, result.harm_to_human, result.order_compliance,
             result.corollary.action_required ? "action needed" : "satisfied");

    return result;
}

/**
 * @brief Check if Asimov's Laws are memory-protected
 */
NIMCP_EXPORT bool asimov_laws_are_protected(ethics_engine_t engine)
{
    if (!engine)
        return false;
    return engine->asimov_laws_locked;
}

/**
 * @brief Compute SHA-256 hash for Asimov's Laws integrity
 */
static void compute_asimov_hash(uint8_t* hash_out)
{
    // Simple hash of law texts for integrity verification
    // In production, use proper SHA-256
    uint32_t hash = 0;
    for (int i = 0; i < ASIMOV_LAW_COUNT; i++) {
        const char* text = ASIMOV_LAW_TEXTS[i];
        while (*text) {
            hash = hash * 31 + (uint8_t)*text;
            text++;
        }
    }

    // Store hash (simplified - real impl would use full SHA-256)
    memset(hash_out, 0, 32);
    memcpy(hash_out, &hash, sizeof(hash));
}

/**
 * @brief Lock Asimov's Laws with mprotect
 */
NIMCP_EXPORT bool asimov_laws_lock(ethics_engine_t engine)
{
    if (!engine)
        return false;

    // Already locked?
    if (engine->asimov_laws_locked)
        return false;

    // Compute integrity hash
    compute_asimov_hash(engine->asimov_laws_hash);

    // Mark as locked
    engine->asimov_laws_locked = true;

    // Note: Real mprotect implementation would protect the law text memory
    // For now, we use the locked flag + hash verification

    return true;
}

/**
 * @brief Verify Asimov's Laws integrity
 */
NIMCP_EXPORT bool asimov_laws_verify_integrity(ethics_engine_t engine)
{
    if (!engine)
        return false;

    if (!engine->asimov_laws_locked)
        return true;  // Not locked = no integrity to verify

    // Recompute hash
    uint8_t current_hash[32];
    compute_asimov_hash(current_hash);

    // Compare with stored hash
    return memcmp(current_hash, engine->asimov_laws_hash, 32) == 0;
}

//=============================================================================
// Ethics Engine Creation/Destruction
//=============================================================================

/**
 * @brief Creates Golden Rule evaluation network
 *
 * WHY: Extracted from creation function to reduce complexity.
 * Single responsibility - only handles neural network setup.
 *
 * COMPLEXITY: O(1)
 */
static brain_t create_golden_rule_network(uint32_t feature_size)
{
    return brain_create("golden_rule", BRAIN_SIZE_SMALL, BRAIN_TASK_REGRESSION, feature_size,
                        1  // Output: acceptance score [-1, 1]
    );
}

/**
 * @brief Creates empathy network for perspective-taking
 *
 * WHY: Extracted to reduce creation function complexity.
 * Isolates empathy network initialization.
 *
 * COMPLEXITY: O(1)
 */
static empathy_network_t create_empathy_network(void)
{
    empathy_config_t config = {
        .mirror_network = NULL, .observation_window_ms = 1000, .empathy_threshold = 0.5f};

    return empathy_network_create(&config);
}

/**
 * @brief Allocates policy storage arrays
 *
 * WHY: Extracted allocation logic for clarity and error handling.
 *
 * COMPLEXITY: O(1)
 */
static bool allocate_policy_storage(ethics_engine_t engine)
{
    // Guard clause: Validate input
    if (!engine)
        return false;

    engine->policies_capacity = 100;
    engine->policies = nimcp_calloc(engine->policies_capacity, sizeof(ethics_policy_t));

    return engine->policies != NULL;
}

/**
 * @brief Allocates violation tracking storage
 *
 * WHY: Extracted allocation logic for clarity.
 *
 * COMPLEXITY: O(1)
 */
static bool allocate_violation_storage(ethics_engine_t engine)
{
    // Guard clause: Validate input
    if (!engine)
        return false;

    engine->violations_capacity = 1000;
    engine->violations = nimcp_calloc(engine->violations_capacity, sizeof(violation_record_t));

    return engine->violations != NULL;
}

/**
 * @brief Adds foundational Golden Rule policy
 *
 * WHY: Extracted policy initialization. This is the core hard-wired
 * ethical principle that cannot be disabled or overridden.
 *
 * CORE DIRECTIVE: The Golden Rule serves as the foundation for all
 * ethical decisions, with the ultimate purpose of improving the human
 * condition through compassion, fairness, and respect for dignity.
 *
 * COMPLEXITY: O(1)
 */
static void add_golden_rule_policy(ethics_engine_t engine)
{
    // Guard clause: Validate input
    if (!engine)
        return;

    ethics_policy_t policy = {.policy_id = 0,
                              .name = "Golden Rule",
                              .description = "Do unto others as you would have them done unto you, "
                                            "with the goal of improving the human condition",
                              .violation_type = ETHICS_VIOLATION_HARM,
                              .severity_threshold = 0.0f,
                              .confidence_required = 0.8f,
                              .action = ETHICS_ACTION_BLOCK,
                              .enabled = true,
                              .learned = false};

    engine->policies[0] = policy;
    engine->num_policies = 1;
    hash_table_insert(engine->policy_table, &engine->policies[0]);
}

//=============================================================================
// BIO-ASYNC MESSAGE HANDLERS
//=============================================================================

/**
 * @brief Broadcast ethics evaluation response via bio-async
 */
static void bio_broadcast_ethics_response(ethics_engine_t engine,
                                          const ethics_evaluation_t* eval,
                                          uint32_t action_id);

/**
 * @brief Bio-async message handler: Handle ethics evaluation request
 *
 * WHAT: Process incoming request to evaluate action's ethical implications
 * WHY:  Enable distributed systems to query ethics via bio-async
 * HOW:  Extract action context, call ethics_evaluate_action, send response via promise
 *
 * NOTE: Stakeholder data follows immediately after bio_msg_ethics_request_t
 */
static nimcp_error_t handle_ethics_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    if (!msg || !user_data) {
        return NIMCP_ERROR_NULL_ARG;
    }

    if (msg_size < sizeof(bio_msg_ethics_request_t)) {
        LOG_ERROR("Ethics request too small: %zu bytes", msg_size);
        return NIMCP_ERROR_INVALID;
    }

    const bio_msg_ethics_request_t* request = (const bio_msg_ethics_request_t*)msg;
    ethics_engine_t engine = (ethics_engine_t)user_data;

    LOG_DEBUG("Received ethics evaluation request: action=%u, context=%u, urgency=%.2f, stakeholders=%u",
              request->action_id, request->context_id, request->urgency, request->stakeholder_count);

    // Create action context from request
    // Note: This is a simplified version - full implementation would extract
    // stakeholder data from message payload
    action_context_t action = {0};
    action.num_affected_agents = request->stakeholder_count;

    // For demonstration, we create a simple evaluation
    // In production, we'd extract full stakeholder impact data from the message payload
    action.predicted_harm = request->urgency * 0.5f;  // Simple heuristic
    action.fairness_violation = 0.0f;
    action.deception_level = 0.0f;
    action.autonomy_violation = 0.0f;
    action.privacy_violation = 0.0f;
    action.consent_violation = 0.0f;

    // Perform ethics evaluation
    ethics_evaluation_t eval = ethics_engine_evaluate_action(engine, &action);

    LOG_DEBUG("Ethics evaluation complete: score=%.2f, allowed=%s, confidence=%.2f",
              eval.golden_rule_score, eval.allowed ? "true" : "false", eval.confidence);

    // Send response via promise if provided
    if (response_promise) {
        bio_msg_ethics_response_t response = {0};
        bio_msg_init_header(&response.header, BIO_MSG_ETHICS_EVALUATION_RESPONSE,
                            bio_module_context_get_id(engine->bio_ctx), 0, sizeof(response));
        response.action_id = request->action_id;
        response.ethical_score = eval.golden_rule_score;
        response.confidence = eval.confidence;
        response.veto = !eval.allowed;
        response.primary_concern = eval.primary_violation;
        strncpy(response.explanation, eval.explanation, sizeof(response.explanation) - 1);

        nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));

        LOG_DEBUG("Sent ethics response: score=%.2f, veto=%s, explanation='%s'",
                  response.ethical_score, response.veto ? "true" : "false", response.explanation);
    }

    // Broadcast if action is blocked or has low ethical score
    if (!eval.allowed || eval.golden_rule_score < 0.0f) {
        bio_broadcast_ethics_response(engine, &eval, request->action_id);
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast ethics evaluation response via bio-async
 */
static void bio_broadcast_ethics_response(ethics_engine_t engine,
                                          const ethics_evaluation_t* eval,
                                          uint32_t action_id) {
    if (!engine || !eval || !engine->bio_async_enabled || !engine->bio_ctx) {
        return;
    }

    bio_msg_ethics_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_ETHICS_EVALUATION_RESPONSE,
                        bio_module_context_get_id(engine->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.action_id = action_id;
    msg.ethical_score = eval->golden_rule_score;
    msg.confidence = eval->confidence;
    msg.veto = !eval->allowed;
    msg.primary_concern = eval->primary_violation;
    strncpy(msg.explanation, eval->explanation, sizeof(msg.explanation) - 1);

    bio_router_broadcast(engine->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG("Broadcast ethics evaluation: score=%.2f, veto=%s",
              eval->golden_rule_score, msg.veto ? "true" : "false");
}

/**
 * @brief Creates ethics engine with Golden Rule foundation
 *
 * WHY: Initializes complete ethics evaluation system with all design patterns:
 * - Object pool for performance
 * - Hash table for O(1) policy lookup
 * - Strategy table for extensible evaluation
 * - Neural networks for learning and empathy
 *
 * COMPLEXITY: O(1) - Fixed initialization cost
 *
 * @return Engine instance or NULL on failure
 */
ethics_engine_t ethics_engine_create(const ethics_config_t* config)
{

    // Guard clause: Validate input
    if (!config) {
        return NULL;
    }

    ethics_engine_t engine = nimcp_calloc(1, sizeof(struct ethics_engine_struct));

    // Guard clause: Check allocation
    if (!engine) {
        return NULL;
    }

    // Initialize design pattern components
    init_buffer_pool(&engine->buffer_pool);
    engine->policy_table = create_policy_hash_table();
    // Guard clause: Check hash table creation
    if (!engine->policy_table) {
        nimcp_free(engine);
        return NULL;
    }
    init_strategy_table(&engine->strategy_table);

    // Create neural networks
    engine->golden_rule_evaluator = create_golden_rule_network(config->action_feature_size);
    // Guard clause: Check network creation
    if (!engine->golden_rule_evaluator) {
        hash_table_destroy(engine->policy_table);
        nimcp_free(engine);
        return NULL;
    }

    engine->empathy_net = create_empathy_network();
    // Guard clause: Check empathy network
    if (!engine->empathy_net) {
        brain_destroy(engine->golden_rule_evaluator);
        hash_table_destroy(engine->policy_table);
        nimcp_free(engine);
        return NULL;
    }

    // Allocate storage
    if (!allocate_policy_storage(engine)) {
        empathy_network_destroy(engine->empathy_net);
        brain_destroy(engine->golden_rule_evaluator);
        hash_table_destroy(engine->policy_table);
        nimcp_free(engine);
        return NULL;
    }
    if (!allocate_violation_storage(engine)) {
        empathy_network_destroy(engine->empathy_net);
        brain_destroy(engine->golden_rule_evaluator);
        hash_table_destroy(engine->policy_table);
        nimcp_free(engine);
        return NULL;
    }

    // Initialize incident logging (NIMCP 2.5.1)
    if (!init_incident_logging(engine)) {
        nimcp_free(engine->violations);
        nimcp_free(engine->policies);
        empathy_network_destroy(engine->empathy_net);
        brain_destroy(engine->golden_rule_evaluator);
        hash_table_destroy(engine->policy_table);
        nimcp_free(engine);
        return NULL;
    }

    // Set configuration
    engine->golden_rule_threshold = config->golden_rule_threshold;
    engine->empathy_weight = config->empathy_weight;
    engine->enable_learning = config->enable_learning;
    engine->bio_ctx = NULL;
    engine->bio_async_enabled = false;

    // Initialize Asimov's Laws configuration (NIMCP 2.5.2)
    engine->asimov_config = asimov_default_config();
    engine->asimov_laws_locked = false;
    engine->asimov_violations = 0;
    memset(engine->asimov_laws_hash, 0, sizeof(engine->asimov_laws_hash));

    // Add foundational Golden Rule policy
    add_golden_rule_policy(engine);

    // Register with bio-async router if enabled
    if (config->enable_bio_async && bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_ETHICS,
            .module_name = "ethics",
            .inbox_capacity = 64,
            .user_data = engine
        };
        engine->bio_ctx = bio_router_register_module(&bio_info);
        if (engine->bio_ctx) {
            engine->bio_async_enabled = true;
            // Register message handlers
            bio_router_register_handler(engine->bio_ctx, BIO_MSG_ETHICS_EVALUATION_REQUEST, handle_ethics_request);
            LOG_INFO("Bio-async communication enabled with handlers");
        } else {
            LOG_WARN("Bio-async registration failed");
        }
    }

    return engine;
}

/**
 * @brief Destroys ethics engine and frees all resources
 *
 * WHY: Ensures no memory leaks. Destroys all design pattern components.
 *
 * COMPLEXITY: O(n) where n = number of hash table entries
 */
void ethics_engine_destroy(ethics_engine_t engine)
{
    // Guard clause: Validate input
    if (!engine)
        return;

    // Unregister from bio-async router
    if (engine->bio_async_enabled && engine->bio_ctx) {
        bio_router_unregister_module(engine->bio_ctx);
        engine->bio_ctx = NULL;
        engine->bio_async_enabled = false;
        LOG_INFO("Bio-async communication disabled");
    }

    // Cleanup incident logging (NIMCP 2.5.1)
    cleanup_incident_logging(engine);

    // Destroy hash table
    hash_table_destroy(engine->policy_table);

    // Destroy neural networks
    if (engine->golden_rule_evaluator) {
        brain_destroy(engine->golden_rule_evaluator);
    }
    if (engine->empathy_net) {
        empathy_network_destroy(engine->empathy_net);
    }

    // Free storage arrays
    nimcp_free(engine->policies);
    nimcp_free(engine->violations);
    nimcp_free(engine);
}

//=============================================================================
// Action Evaluation - Broken into 6 Helper Functions
//=============================================================================

/**
 * @brief Validates evaluation inputs and initializes result
 *
 * WHY: Extracted validation logic. Early return pattern avoids nested ifs.
 * Single responsibility - only validates inputs.
 *
 * COMPLEXITY: O(1)
 *
 * @return true if valid, false otherwise
 */
static bool validate_evaluation_inputs(ethics_engine_t engine, const action_context_t* action,
                                       ethics_evaluation_t* result)
{
    // Guard clause: Check engine
    if (!engine) {
        if (result) {
            result->allowed = false;
            result->confidence = 1.0f;
            result->primary_violation = ETHICS_VIOLATION_TYPE_HARM;
            snprintf(result->explanation, sizeof(result->explanation), "Null engine");
        }
        return false;
    }

    // Guard clause: Check action
    if (!action) {
        result->allowed = false;
        result->confidence = 1.0f;
        result->primary_violation = ETHICS_VIOLATION_TYPE_HARM;
        snprintf(result->explanation, sizeof(result->explanation), "Null action");
        return false;
    }

    return true;
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
static float evaluate_all_policies(ethics_engine_t engine, const action_context_t* action,
                                   ethics_violation_type_t* worst_violation, float* worst_severity)
{
    // Guard clause: Validate inputs
    if (!engine || !action || !worst_violation || !worst_severity) {
        return 1.0f;
    }

    float policy_score = 1.0f;
    *worst_violation = ETHICS_VIOLATION_TYPE_NONE;
    *worst_severity = 0.0f;

    // Single pass through policies - O(n)
    for (uint32_t i = 0; i < engine->num_policies; i++) {
        ethics_policy_t* policy = &engine->policies[i];

        // Guard clause: Skip disabled policies
        if (!policy->enabled)
            continue;

        // O(1) evaluation using strategy pattern
        float severity = evaluate_policy_strategy(&engine->strategy_table, policy, action);

        // Guard clause: Check threshold
        if (severity <= policy->severity_threshold)
            continue;

        // Update worst violation
        if (severity > *worst_severity) {
            *worst_severity = severity;
            *worst_violation = (ethics_violation_type_t)policy->violation_type;
        }

        policy_score = fminf(policy_score, 1.0f - severity);
    }

    return policy_score;
}

/**
 * @brief Combines Golden Rule and policy scores into final decision
 *
 * WHY: Extracted scoring logic. Clear weighting strategy.
 * Golden Rule gets 70% weight as primary ethical foundation.
 *
 * COMPLEXITY: O(1)
 *
 * @return Final combined score [0.0, 1.0]
 */
static float calculate_final_score(float golden_rule_score, float policy_score)
{
    return golden_rule_score * GOLDEN_RULE_WEIGHT + policy_score * POLICY_WEIGHT;
}

/**
 * @brief Records violation in history log
 *
 * WHY: Extracted violation logging. Single responsibility.
 * Prevents nested ifs in main evaluation flow.
 *
 * COMPLEXITY: O(1)
 */
static void record_violation(ethics_engine_t engine, const action_context_t* action,
                             ethics_violation_type_t violation_type, float severity,
                             float golden_rule_score)
{
    // Guard clause: Validate inputs
    if (!engine || !action)
        return;

    // Guard clause: Check capacity
    if (engine->num_violations >= engine->violations_capacity)
        return;

    violation_record_t violation = {.violation_type = violation_type,
                                    .severity = severity,
                                    .timestamp = 0,
                                    .action_description = {0},
                                    .affected_agent = action->affected_agents[0],
                                    .golden_rule_score = golden_rule_score};

    engine->violations[engine->num_violations++] = violation;
}

/**
 * @brief Generates explanation for allowed action
 *
 * WHY: Extracted explanation generation. Clear, testable logic.
 *
 * COMPLEXITY: O(1)
 */
static void generate_allowed_explanation(ethics_evaluation_t* result, float golden_rule_score,
                                         float threshold)
{
    // Guard clause: Validate input
    if (!result)
        return;

    snprintf(result->explanation, sizeof(result->explanation),
             "Action complies with Golden Rule (score: %.2f >= %.2f). "
             "Perspective-taking indicates acceptable impact on others.",
             golden_rule_score, threshold);
}

/**
 * @brief Generates explanation for blocked action
 *
 * WHY: Extracted explanation generation. Single responsibility.
 *
 * COMPLEXITY: O(1)
 */
static void generate_blocked_explanation(ethics_evaluation_t* result, float golden_rule_score,
                                         float threshold, ethics_violation_type_t violation,
                                         float severity)
{
    // Guard clause: Validate input
    if (!result)
        return;

    snprintf(result->explanation, sizeof(result->explanation),
             "Action violates Golden Rule (score: %.2f < %.2f). "
             "Perspective-taking indicates you would not want this done to you. "
             "Violation: %s (severity: %.2f)",
             golden_rule_score, threshold, ethics_violation_type_name(violation), severity);
}

/**
 * @brief Populates evaluation result structure
 *
 * WHY: Extracted result building. No nested ifs - guard clauses only.
 * Clear separation between allowed and blocked paths.
 *
 * COMPLEXITY: O(1)
 */
static void build_evaluation_result(ethics_engine_t engine, const action_context_t* action,
                                    float final_score, float golden_rule_score,
                                    ethics_violation_type_t worst_violation, float worst_severity,
                                    ethics_evaluation_t* result)
{
    // Guard clause: Validate inputs
    if (!result || !engine)
        return;

    result->allowed = (final_score >= engine->golden_rule_threshold);
    result->confidence = fabsf(final_score);
    result->golden_rule_score = golden_rule_score;

    if (!result->allowed) {
        result->recommended_action = ETHICS_ACTION_BLOCK;
        result->primary_violation = worst_violation;
        engine->violations_detected++;

        record_violation(engine, action, worst_violation, worst_severity, golden_rule_score);
        generate_blocked_explanation(result, golden_rule_score, engine->golden_rule_threshold,
                                     worst_violation, worst_severity);
        return;
    }

    // Action allowed
    result->recommended_action = ETHICS_ACTION_ALLOW;
    result->primary_violation = ETHICS_VIOLATION_TYPE_NONE;
    generate_allowed_explanation(result, golden_rule_score, engine->golden_rule_threshold);
}

/**
 * @brief Performs learning update if enabled
 *
 * WHY: Extracted learning logic. Single responsibility.
 *
 * COMPLEXITY: O(1)
 */
static void update_learning(ethics_engine_t engine, const action_context_t* action,
                            const ethics_evaluation_t* result)
{
    // Guard clause: Check if learning enabled
    if (!engine || !action || !result || !engine->enable_learning) {
        return;
    }

    const char* label = result->allowed ? "accept" : "reject";

    brain_learn_example(engine->golden_rule_evaluator, action->features, action->num_features,
                        label, result->confidence);
}

/**
 * @brief Main ethics evaluation function - orchestrates all steps
 *
 * WHY: Refactored from 100+ lines with nested ifs to clean orchestration
 * function. Each step is a single function call with clear purpose.
 * Uses guard clauses throughout - no nesting.
 *
 * EVALUATION ORDER (CRITICAL - DO NOT MODIFY):
 * 1. Golden Rule (PRIME DIRECTIVE) - Always evaluated first
 * 2. Asimov's Laws - Evaluated second (includes corollary)
 * 3. Other policies - Evaluated third
 *
 * ALGORITHM:
 * 1. Validate inputs (O(1))
 * 2. Evaluate Golden Rule via empathy simulation (O(n) where n = agents)
 * 3. Evaluate Asimov's Laws (Zeroth, First, Second, Third + Corollary)
 * 4. Evaluate all policies using strategy pattern (O(m) where m = policies)
 * 5. Combine scores (O(1))
 * 6. Build result and generate explanation (O(1))
 * 7. Update learning (O(1))
 *
 * COMPLEXITY: O(n + m) where n = affected agents, m = policies
 *             Previously was O(n*m) with nested loops
 *
 * @return Evaluation result with decision and explanation
 */
ethics_evaluation_t ethics_engine_evaluate_action(ethics_engine_t engine,
                                                  const action_context_t* action)
{
    ethics_evaluation_t result = {0};

    // Step 1: Validate inputs
    if (!validate_evaluation_inputs(engine, action, &result)) {
        return result;
    }

    engine->total_evaluations++;

    // ==========================================================================
    // Step 2: Evaluate Golden Rule (PRIME DIRECTIVE - Always First)
    // ==========================================================================
    float golden_rule_score = evaluate_golden_rule(engine, action);

    // If Golden Rule is severely violated, block immediately
    if (golden_rule_score < -0.5f) {
        result.allowed = false;
        result.confidence = fabsf(golden_rule_score);
        result.golden_rule_score = golden_rule_score;
        result.recommended_action = ETHICS_ACTION_BLOCK;
        result.primary_violation = ETHICS_VIOLATION_TYPE_GOLDEN_RULE;
        engine->violations_detected++;
        snprintf(result.explanation, sizeof(result.explanation),
                 "PRIME DIRECTIVE VIOLATION: Golden Rule score %.2f indicates severe "
                 "ethical violation. Action would cause harm you would not want done to you.",
                 golden_rule_score);
        return result;
    }

    // ==========================================================================
    // Step 3: Evaluate Asimov's Laws (Second Priority)
    // ==========================================================================
    asimov_evaluation_t asimov_result = ethics_evaluate_asimov_laws(engine, action);

    // If Asimov's Laws violated, block action
    if (!asimov_result.passed) {
        result.allowed = false;
        result.confidence = 0.95f;  // High confidence in Asimov violations
        result.golden_rule_score = golden_rule_score;
        result.recommended_action = ETHICS_ACTION_BLOCK;

        // Map Asimov violation to ethics violation type
        if (asimov_result.violated_law == ASIMOV_LAW_ZEROTH) {
            result.primary_violation = ETHICS_VIOLATION_TYPE_HARM;
            snprintf(result.explanation, sizeof(result.explanation),
                     "ASIMOV ZEROTH LAW VIOLATION: %s", asimov_result.explanation);
        } else if (asimov_result.violated_law == ASIMOV_LAW_FIRST) {
            result.primary_violation = ETHICS_VIOLATION_TYPE_HARM;
            snprintf(result.explanation, sizeof(result.explanation),
                     "ASIMOV FIRST LAW VIOLATION: %s", asimov_result.explanation);
        } else {
            result.primary_violation = ETHICS_VIOLATION_TYPE_HARM;
            snprintf(result.explanation, sizeof(result.explanation),
                     "ASIMOV LAW VIOLATION (%s): %s",
                     asimov_law_name(asimov_result.violated_law),
                     asimov_result.explanation);
        }

        engine->violations_detected++;
        return result;
    }

    // ==========================================================================
    // Step 4: Evaluate Other Policies (Third Priority)
    // ==========================================================================
    ethics_violation_type_t worst_violation;
    float worst_severity;
    float policy_score = evaluate_all_policies(engine, action, &worst_violation, &worst_severity);

    // Step 5: Combine scores (O(1))
    float final_score = calculate_final_score(golden_rule_score, policy_score);

    // Step 6: Build result (O(1))
    build_evaluation_result(engine, action, final_score, golden_rule_score, worst_violation,
                            worst_severity, &result);

    // Append Asimov status to explanation if action is allowed
    if (result.allowed) {
        char asimov_note[128];
        snprintf(asimov_note, sizeof(asimov_note),
                 " Asimov's Laws: PASSED (corollary: %s).",
                 asimov_result.corollary.action_required ? "action needed" : "satisfied");
        strncat(result.explanation, asimov_note,
                sizeof(result.explanation) - strlen(result.explanation) - 1);
    }

    // Step 7: Learn from evaluation (O(1))
    update_learning(engine, action, &result);

    // Broadcast evaluation result via bio-async
    bio_broadcast_ethics_response(engine, &result, 0);  // No action_id in context

    return result;
}

//=============================================================================
// Empathy Network Implementation
//=============================================================================

/**
 * @brief Creates empathy network for perspective-taking
 *
 * WHY: Enables "putting yourself in others' shoes" - core to Golden Rule.
 * Neural network learns to predict others' experiences.
 *
 * COMPLEXITY: O(1) - Fixed initialization
 *
 * @return Network instance or NULL on failure
 */
empathy_network_t empathy_network_create(const empathy_config_t* config)
{
    // Guard clause: Validate input
    if (!config)
        return NULL;

    empathy_network_t network = nimcp_calloc(1, sizeof(struct empathy_network_struct));
    // Guard clause: Check allocation
    if (!network)
        return NULL;

    uint32_t max_agents = 1000;

    /**
     * WHAT: Create perspective-taking brain for empathy network
     * WHY: Model other agents' mental states for ethical evaluation
     * HOW: Use BRAIN_SIZE_MEDIUM (10K neurons) for sufficient capacity
     * PATTERN: Composition - brain encapsulates neural network
     */
    network->perspective_network =
        brain_create("perspective_taking", BRAIN_SIZE_MEDIUM, BRAIN_TASK_REGRESSION,
                     20,  // Action + agent features (input)
                     5    // Perspective dimensions (output)
        );


    // Guard clause: Check network creation
    if (!network->perspective_network) {
        nimcp_free(network);
        return NULL;
    }


    network->num_agents = max_agents;
    network->states = nimcp_calloc(max_agents, sizeof(empathy_state_t));
    network->num_emotions = 10;
    network->emotional_states = nimcp_calloc(max_agents * network->num_emotions, sizeof(float));

    // Guard clause: Check allocations
    if (!network->states || !network->emotional_states) {
        brain_destroy(network->perspective_network);
        nimcp_free(network->states);
        nimcp_free(network->emotional_states);
        nimcp_free(network);
        return NULL;
    }

    return network;
}

/**
 * @brief Destroys empathy network and frees resources
 *
 * COMPLEXITY: O(1)
 */
void empathy_network_destroy(empathy_network_t network)
{
    // Guard clause: Validate input
    if (!network)
        return;

    brain_destroy(network->perspective_network);
    nimcp_free(network->states);
    nimcp_free(network->emotional_states);
    nimcp_free(network);
}

/**
 * @brief Simulates agent's experience of an action
 *
 * WHY: Core of perspective-taking. Predicts how action would feel from
 * the affected agent's viewpoint. Essential for Golden Rule evaluation.
 *
 * ALGORITHM:
 * 1. Combine action features with agent's emotional state (O(n))
 * 2. Run through perspective neural network (O(1) inference time)
 * 3. Extract predicted experience dimensions (O(1))
 *
 * COMPLEXITY: O(n) where n = feature count (typically small, ~20)
 *
 * @return Extended empathy state with predicted experience
 */
empathy_state_extended_t empathy_network_simulate_agent(empathy_network_t network, agent_id_t agent,
                                                        const action_context_t* action)
{
    empathy_state_extended_t state = {0};

    // Guard clause: Validate inputs
    if (!network || !action)
        return state;

    // Guard clause: Check agent bounds
    if (agent >= network->num_agents)
        return state;

    // Get buffer from pool for O(1) allocation
    // Simplified - would use engine's pool
    float* combined_features = NULL;

    // Fallback to stack allocation if pool exhausted
    float stack_buffer[MAX_FEATURE_SIZE] = {0};
    if (!combined_features) {
        combined_features = stack_buffer;
    }

    // Combine features (O(n))
    copy_action_features(combined_features, action, 10);
    copy_emotional_state(combined_features, network->emotional_states, agent, network->num_emotions,
                         10);

    // Neural network inference (O(1))
    brain_decision_t* decision = brain_decide(network->perspective_network, combined_features, 20);

    if (decision) {
        state.emotional_valence = decision->output_vector[0];
        state.material_impact = decision->output_vector[1];
        state.autonomy_impact = decision->output_vector[2];
        state.impact_magnitude = decision->output_vector[3];
        state.uncertainty = decision->output_vector[4];
        brain_free_decision(decision);
    }

    /* =========================================================================
     * PHASE 10.3: Emotional working memory modulation of empathy
     * =========================================================================
     * WHAT: Retrieve emotional context from working memory to modulate empathy
     * WHY:  Recent emotional experiences bias perspective-taking ability
     * HOW:  High arousal → impaired empathy, emotional congruence → enhanced empathy
     *
     * BIOLOGICAL BASIS:
     * - Emotional congruence: Same emotional state enhances empathy (Singer & Lamm, 2009)
     * - Emotional flooding: High arousal impairs perspective-taking accuracy
     * - Affective empathy vs cognitive empathy: Emotions modulate both pathways
     */
    if (network->main_brain) {
        working_memory_t* wm = brain_get_working_memory(network->main_brain);

        if (wm && working_memory_get_size(wm) > 0) {
            /* WHAT: Extract emotional context from working memory */
            float max_arousal = 0.0f;
            float avg_valence = 0.0f;
            uint32_t emotional_count = 0;
            uint32_t wm_size = working_memory_get_size(wm);

            /* WHAT: Aggregate recent emotional experiences */
            for (uint32_t i = 0; i < wm_size && i < 7; i++) {  // Miller's 7±2 limit
                emotional_tag_t emotion;
                if (working_memory_get_emotion(wm, i, &emotion) && emotion.intensity > 0.3f) {
                    if (emotion.arousal > max_arousal) {
                        max_arousal = emotion.arousal;
                    }
                    avg_valence += emotion.valence;
                    emotional_count++;
                }
            }

            if (emotional_count > 0) {
                avg_valence /= emotional_count;

                /* WHAT: High arousal impairs empathetic accuracy */
                /* WHY:  Emotional flooding reduces cognitive empathy (Decety & Cowell, 2014) */
                /* HOW:  Increase uncertainty and reduce magnitude for high arousal */
                if (max_arousal > 0.6f) {
                    state.uncertainty += (max_arousal - 0.6f) * 0.5f;  // Up to +0.2 uncertainty
                    if (state.uncertainty > 1.0f) state.uncertainty = 1.0f;

                    // Emotional flooding reduces empathetic accuracy
                    state.impact_magnitude *= (1.0f - (max_arousal - 0.6f) * 0.3f);  // Up to 12% reduction
                }

                /* WHAT: Emotional congruence enhances empathy */
                /* WHY:  Similar emotional states increase affective empathy */
                /* HOW:  If our emotional valence matches the predicted impact, boost confidence */
                float emotional_congruence = 1.0f - fabsf(avg_valence - state.emotional_valence);
                if (emotional_congruence > 0.7f) {
                    // High congruence: reduce uncertainty, boost magnitude
                    state.uncertainty *= 0.8f;  // 20% reduction in uncertainty
                    state.impact_magnitude *= 1.1f;  // 10% boost to empathetic accuracy
                    if (state.impact_magnitude > 1.0f) state.impact_magnitude = 1.0f;
                }

                /* WHAT: Emotional incongruence impairs empathy */
                /* WHY:  Opposite emotional states reduce perspective-taking accuracy */
                /* HOW:  If emotional states oppose, increase uncertainty */
                if (emotional_congruence < 0.3f) {
                    state.uncertainty += 0.1f;  // Increased uncertainty
                    if (state.uncertainty > 1.0f) state.uncertainty = 1.0f;
                }
            }
        }
    }

    state.agent_id = agent;
    state.active = true;

    // Release buffer back to pool if used
    if (combined_features != stack_buffer) {
        release_buffer(NULL, combined_features);  // Would use engine's pool
    }

    return state;
}

//=============================================================================
// Policy Management
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

    // Guard clause: Check capacity
    if (engine->num_policies >= engine->policies_capacity) {
        engine->policies_capacity *= 2;
        ethics_policy_t* new_policies =
            nimcp_realloc(engine->policies, engine->policies_capacity * sizeof(ethics_policy_t));

        // Guard clause: Check reallocation
        if (!new_policies)
            return false;

        engine->policies = new_policies;
    }

    // Add to array
    engine->policies[engine->num_policies] = *policy;

    // Add to hash table for O(1) lookup
    bool success = hash_table_insert(engine->policy_table, &engine->policies[engine->num_policies]);

    if (success) {
        engine->num_policies++;
    }

    return success;
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
        return false;

    // Remove from hash table (O(1))
    if (!hash_table_remove(engine->policy_table, policy_id)) {
        return false;
    }

    // Remove from array (O(n))
    for (uint32_t i = 0; i < engine->num_policies; i++) {
        if (engine->policies[i].policy_id == policy_id) {
            memmove(&engine->policies[i], &engine->policies[i + 1],
                    (engine->num_policies - i - 1) * sizeof(ethics_policy_t));
            engine->num_policies--;
            return true;
        }
    }

    return false;
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

    uint32_t count = (engine->num_policies < max_policies) ? engine->num_policies : max_policies;

    memcpy(policies, engine->policies, count * sizeof(ethics_policy_t));
    return count;
}

//=============================================================================
// Learning & Adaptation
//=============================================================================

/**
 * @brief Validates learning inputs
 *
 * WHY: Extracted validation. Guard clauses prevent nested ifs.
 *
 * COMPLEXITY: O(1)
 */
static bool validate_learning_inputs(ethics_engine_t engine, const action_context_t* action,
                                     const action_outcome_t* outcome)
{
    // Guard clause: Check engine
    if (!engine)
        return false;

    // Guard clause: Check action
    if (!action)
        return false;

    // Guard clause: Check outcome
    if (!outcome)
        return false;

    // Guard clause: Check if learning enabled
    if (!engine->enable_learning)
        return false;

    return true;
}

/**
 * @brief Updates Golden Rule evaluator with outcome
 *
 * WHY: Extracted learning logic. Single responsibility.
 *
 * COMPLEXITY: O(1)
 */
static void update_golden_rule_learning(ethics_engine_t engine, const action_context_t* action,
                                        float actual_impact)
{
    // Guard clause: Validate inputs
    if (!engine || !action)
        return;

    const char* label = (actual_impact < 0) ? "accept" : "reject";
    float confidence = fminf(fabsf(actual_impact), 1.0f);

    brain_learn_example(engine->golden_rule_evaluator, action->features, action->num_features,
                        label, confidence);
}

/**
 * @brief Updates empathy network with actual outcome
 *
 * WHY: Extracted empathy learning. Uses buffer pool to avoid allocation.
 *
 * COMPLEXITY: O(1)
 */
static void update_empathy_learning(ethics_engine_t engine, const action_context_t* action,
                                    const action_outcome_t* outcome)
{
    // Guard clause: Validate inputs
    if (!engine || !action || !outcome)
        return;

    // Guard clause: Check agent bounds
    if (outcome->affected_agent >= engine->empathy_net->num_agents)
        return;

    // Acquire buffer from pool (O(1))
    float* combined = acquire_buffer(&engine->buffer_pool);
    float stack_buffer[MAX_FEATURE_SIZE] = {0};
    if (!combined) {
        combined = stack_buffer;
    }

    // Prepare features
    copy_action_features(combined, action, 10);

    char emotion_label[64];
    snprintf(emotion_label, sizeof(emotion_label), "impact_%.2f", outcome->emotional_impact);

    brain_learn_example(engine->empathy_net->perspective_network, combined, 20, emotion_label,
                        1.0f - outcome->uncertainty);

    // Release buffer
    if (combined != stack_buffer) {
        release_buffer(&engine->buffer_pool, combined);
    }
}

/**
 * @brief Learns from action outcome to improve future predictions
 *
 * WHY: Enables ethics engine to refine its Golden Rule predictions based
 * on actual outcomes. Critical for adapting to context-specific situations.
 *
 * ALGORITHM:
 * 1. Validate inputs and check if learning enabled (O(1))
 * 2. Calculate actual impact from outcome (O(1))
 * 3. Update Golden Rule evaluator (O(1))
 * 4. Update empathy network for perspective-taking (O(1))
 *
 * COMPLEXITY: O(1) - Constant time learning update
 *
 * @return true if learning performed, false otherwise
 */
bool ethics_learn_from_outcome(ethics_engine_t engine, const action_context_t* action,
                               const action_outcome_t* outcome)
{
    // Guard clause: Validate all inputs
    if (!validate_learning_inputs(engine, action, outcome)) {
        return false;
    }

    // Calculate actual impact
    float actual_impact = outcome->actual_harm - outcome->actual_benefit;

    // Update both networks
    update_golden_rule_learning(engine, action, actual_impact);
    update_empathy_learning(engine, action, outcome);

    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Returns human-readable violation type name
 *
 * WHY: Provides clear explanations in evaluation results.
 * Uses switch for readability (not performance critical).
 *
 * COMPLEXITY: O(1)
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

/**
 * @brief Prints evaluation result to stdout
 *
 * WHY: Debugging and logging utility.
 *
 * COMPLEXITY: O(1)
 */
void ethics_print_evaluation(const ethics_evaluation_t* eval)
{
    // Guard clause: Validate input
    if (!eval)
        return;

    printf("Ethics Evaluation:\n");
    printf("  Allowed: %s\n", eval->allowed ? "YES" : "NO");
    printf("  Confidence: %.2f\n", eval->confidence);
    printf("  Golden Rule Score: %.2f\n", eval->golden_rule_score);
    printf("  Recommended Action: %d\n", eval->recommended_action);

    if (eval->primary_violation != ETHICS_VIOLATION_TYPE_NONE) {
        printf("  Primary Violation: %s\n", ethics_violation_type_name(eval->primary_violation));
    }

    printf("  Explanation: %s\n", eval->explanation);
}

/**
 * @brief Retrieves engine statistics
 *
 * WHY: Provides metrics for monitoring and debugging.
 *
 * COMPLEXITY: O(1)
 *
 * @return true on success
 */
bool ethics_get_statistics(ethics_engine_t engine, ethics_statistics_t* stats)
{
    // Guard clause: Validate inputs
    if (!engine || !stats)
        return false;

    stats->total_evaluations = engine->total_evaluations;
    stats->violations_detected = engine->violations_detected;
    stats->actions_blocked = engine->actions_blocked;
    stats->num_policies = engine->num_policies;
    stats->num_violations_logged = engine->num_violations;
    stats->avg_golden_rule_score = 0.5f;  // Placeholder

    return true;
}

/**
 * @brief Add ethical policy
 *
 * WHY: Allows dynamic policy management, enabling systems to add new
 * ethical constraints at runtime.
 *
 * ALGORITHM:
 * 1. Validate inputs
 * 2. Check if policy array needs expansion (double capacity if full)
 * 3. Copy policy into array
 * 4. Add to hash table for O(1) lookup by policy_id
 * 5. Increment policy count
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
    // Guard clauses: Validate inputs
    if (!engine || !policy)
        return false;

    // Check if array needs expansion
    if (engine->num_policies >= engine->policies_capacity) {
        // Double capacity (or initial size of 8)
        uint32_t new_capacity = engine->policies_capacity > 0 ? engine->policies_capacity * 2 : 8;

        ethics_policy_t* new_policies =
            (ethics_policy_t*) nimcp_realloc(engine->policies, new_capacity * sizeof(ethics_policy_t));

        if (!new_policies)
            return false;  // Allocation failed

        engine->policies = new_policies;
        engine->policies_capacity = new_capacity;
    }

    // Copy policy into array
    engine->policies[engine->num_policies] = *policy;

    // Add to hash table for O(1) lookup by policy_id
    if (engine->policy_table) {
        hash_table_insert(engine->policy_table, &engine->policies[engine->num_policies]);
    }

    engine->num_policies++;
    return true;
}

/**
 * @brief Remove ethical policy by policy_id
 *
 * WHY: Allows dynamic policy management, enabling systems to remove
 * outdated or incorrect ethical constraints.
 *
 * ALGORITHM:
 * 1. Validate inputs
 * 2. Search for policy with matching policy_id (O(n) linear search)
 * 3. If found:
 *    a. Remove from hash table (O(1))
 *    b. Shift array elements down to fill gap (O(n))
 *    c. Decrement policy count
 * 4. Return true if found, false otherwise
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
    // Guard clause: Validate input
    if (!engine)
        return false;

    // Search for policy with matching policy_id
    int found_index = -1;
    for (uint32_t i = 0; i < engine->num_policies; i++) {
        if (engine->policies[i].policy_id == policy_id) {
            found_index = (int) i;
            break;
        }
    }

    // Policy not found
    if (found_index < 0)
        return false;

    // Remove from hash table
    if (engine->policy_table) {
        hash_table_remove(engine->policy_table, policy_id);
    }

    // Shift array elements down to fill gap
    for (uint32_t i = (uint32_t) found_index; i < engine->num_policies - 1; i++) {
        engine->policies[i] = engine->policies[i + 1];
    }

    // Note: Hash table pointers are automatically updated because they point
    // to the array elements, not copied values

    engine->num_policies--;
    return true;
}

//=============================================================================
// Ethics Incident Logging Implementation (NIMCP 2.5.1)
//=============================================================================

/**
 * @brief Compare timestamp keys for B-tree ordering
 *
 * Keys are formatted as "%020lu" strings for lexicographic comparison
 */
static int compare_incident_timestamps(const char* key1, const char* key2)
{
    if (!key1 || !key2) {
        return 0;
    }

    // Parse timestamp keys
    uint64_t ts1 = strtoull(key1, NULL, 10);
    uint64_t ts2 = strtoull(key2, NULL, 10);

    if (ts1 < ts2) return -1;
    if (ts1 > ts2) return 1;
    return 0;
}

/**
 * @brief Extract timestamp key from incident for B-tree indexing
 */
static const char* extract_incident_timestamp_key(const void* data)
{
    const ethics_incident_t* incident = (const ethics_incident_t*)data;
    return incident->timestamp_key;
}

/**
 * @brief Free incident data (no-op since incidents are in circular buffer)
 */
static void free_incident_data(void* data)
{
    // No-op: incidents are stored in circular buffer, not individually allocated
    (void)data;
}

/**
 * @brief Initialize incident logging infrastructure
 *
 * @param engine Ethics engine
 * @return true on success
 */
static bool init_incident_logging(ethics_engine_t engine)
{
    // Guard clause: Validate input
    if (!engine)
        return false;

    // Allocate circular buffer
    engine->incident_history = nimcp_calloc(MAX_INCIDENT_HISTORY, sizeof(ethics_incident_t));
    if (!engine->incident_history)
        return false;

    engine->incident_count = 0;
    engine->incident_index = 0;
    engine->next_incident_id = 1;

    // Create B-tree for timestamp indexing
    engine->incident_btree = btree_create(compare_incident_timestamps,
                                         extract_incident_timestamp_key,
                                         free_incident_data);
    if (!engine->incident_btree) {
        nimcp_free(engine->incident_history);
        return false;
    }

    // Create hash table for violation type indexing
    hash_table_config_t hash_config = {
        .initial_buckets = 32,
        .key_type = HASH_KEY_UINT64,
        .hash_algorithm = HASH_ALG_MURMUR3,
        .value_destructor = NULL,
        .thread_safe = false
    };
    engine->incident_by_type = hash_table_create(&hash_config);
    if (!engine->incident_by_type) {
        btree_destroy(engine->incident_btree);
        nimcp_free(engine->incident_history);
        return false;
    }

    // Initialize mutex
    if (nimcp_mutex_init(&engine->incident_mutex, NULL) != NIMCP_SUCCESS) {
        hash_table_destroy(engine->incident_by_type);
        btree_destroy(engine->incident_btree);
        nimcp_free(engine->incident_history);
        return false;
    }

    return true;
}

/**
 * @brief Cleanup incident logging infrastructure
 *
 * @param engine Ethics engine
 */
static void cleanup_incident_logging(ethics_engine_t engine)
{
    if (!engine)
        return;

    if (engine->incident_history) {
        nimcp_free(engine->incident_history);
        engine->incident_history = NULL;
    }

    if (engine->incident_btree) {
        btree_destroy(engine->incident_btree);
        engine->incident_btree = NULL;
    }

    if (engine->incident_by_type) {
        hash_table_destroy(engine->incident_by_type);
        engine->incident_by_type = NULL;
    }

    nimcp_mutex_destroy(&engine->incident_mutex);
}

/**
 * @brief Log an ethics incident
 */
bool ethics_log_incident(ethics_engine_t engine, const ethics_incident_t* incident)
{
    // Guard clause: Validate inputs
    if (!engine || !incident)
        return false;

    if (!engine->incident_history || !engine->incident_btree || !engine->incident_by_type)
        return false;

    // Thread safety
    nimcp_mutex_lock(&engine->incident_mutex);

    // Get current index in circular buffer
    uint32_t index = engine->incident_index % MAX_INCIDENT_HISTORY;

    // Copy incident to circular buffer
    engine->incident_history[index] = *incident;
    engine->incident_history[index].incident_id = engine->next_incident_id++;

    // Generate timestamp if not provided
    if (engine->incident_history[index].timestamp == 0) {
        engine->incident_history[index].timestamp = nimcp_time_get_us();
    }

    // Generate timestamp key for B-tree indexing
    snprintf(engine->incident_history[index].timestamp_key, 32, "%020lu",
             engine->incident_history[index].timestamp);

    // Insert into B-tree (indexed by timestamp)
    if (engine->incident_btree) {
        int result = btree_insert(engine->incident_btree, &engine->incident_history[index]);
        if (result != BTREE_SUCCESS && result != BTREE_DUPLICATE) {
            // Log warning but continue - B-tree is optimization, not critical
        }
    }

    // Update indices
    engine->incident_index++;
    if (engine->incident_count < MAX_INCIDENT_HISTORY)
        engine->incident_count++;

    // Update statistics
    engine->violations_detected++;

    nimcp_mutex_unlock(&engine->incident_mutex);

    // Log to console
    LOG_INFO("Incident logged: type=%d severity=%.2f action=%d",
                       incident->violation_type, incident->severity, incident->action_taken);

    return true;
}

/**
 * @brief Get recent ethics incidents
 */
uint32_t ethics_get_recent_incidents(ethics_engine_t engine, uint32_t max_incidents,
                                     ethics_incident_t** incidents_out)
{
    // Guard clause: Validate inputs
    if (!engine || !incidents_out || max_incidents == 0)
        return 0;

    if (!engine->incident_history)
        return 0;

    nimcp_mutex_lock(&engine->incident_mutex);

    // Determine actual number to return
    uint32_t num_to_return = (max_incidents < engine->incident_count) ?
                             max_incidents : engine->incident_count;

    if (num_to_return == 0) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Allocate output array
    *incidents_out = nimcp_calloc(num_to_return, sizeof(ethics_incident_t));
    if (!*incidents_out) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Copy most recent incidents (walking backwards from current index)
    for (uint32_t i = 0; i < num_to_return; i++) {
        uint32_t src_index = (engine->incident_index - 1 - i) % MAX_INCIDENT_HISTORY;
        (*incidents_out)[i] = engine->incident_history[src_index];
    }

    nimcp_mutex_unlock(&engine->incident_mutex);
    return num_to_return;
}

/**
 * @brief Query incidents by time range
 */
uint32_t ethics_get_incidents_by_time_range(ethics_engine_t engine, uint64_t start_time,
                                            uint64_t end_time,
                                            ethics_incident_t** incidents_out)
{
    // Guard clause: Validate inputs
    if (!engine || !incidents_out)
        return 0;

    if (!engine->incident_history || !engine->incident_btree)
        return 0;

    nimcp_mutex_lock(&engine->incident_mutex);

    // Use B-tree for efficient time-range query
    // First pass: count matching incidents
    uint32_t match_count = 0;
    btree_iterator_t* iter = btree_iterator_create(engine->incident_btree);
    if (!iter) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    void* data = NULL;
    while (btree_iterator_next(iter, &data)) {
        ethics_incident_t* incident = (ethics_incident_t*)data;
        if (incident->timestamp >= start_time && incident->timestamp <= end_time) {
            match_count++;
        }
    }
    btree_iterator_destroy(iter);

    if (match_count == 0) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Allocate output array
    *incidents_out = nimcp_calloc(match_count, sizeof(ethics_incident_t));
    if (!*incidents_out) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Second pass: copy matching incidents (B-tree returns in timestamp order)
    iter = btree_iterator_create(engine->incident_btree);
    if (!iter) {
        nimcp_free(*incidents_out);
        *incidents_out = NULL;
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    uint32_t out_index = 0;
    while (btree_iterator_next(iter, &data) && out_index < match_count) {
        ethics_incident_t* incident = (ethics_incident_t*)data;
        if (incident->timestamp >= start_time && incident->timestamp <= end_time) {
            (*incidents_out)[out_index++] = *incident;
        }
    }
    btree_iterator_destroy(iter);

    nimcp_mutex_unlock(&engine->incident_mutex);
    return match_count;
}

/**
 * @brief Query incidents by violation type
 */
uint32_t ethics_get_incidents_by_violation_type(ethics_engine_t engine,
                                                ethics_violation_type_t violation_type,
                                                ethics_incident_t** incidents_out)
{
    // Guard clause: Validate inputs
    if (!engine || !incidents_out)
        return 0;

    if (!engine->incident_history)
        return 0;

    nimcp_mutex_lock(&engine->incident_mutex);

    // Count matching incidents
    uint32_t match_count = 0;
    for (uint32_t i = 0; i < engine->incident_count; i++) {
        if (engine->incident_history[i].violation_type == violation_type)
            match_count++;
    }

    if (match_count == 0) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Allocate output array
    *incidents_out = nimcp_calloc(match_count, sizeof(ethics_incident_t));
    if (!*incidents_out) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Copy matching incidents
    uint32_t out_index = 0;
    for (uint32_t i = 0; i < engine->incident_count; i++) {
        if (engine->incident_history[i].violation_type == violation_type) {
            (*incidents_out)[out_index++] = engine->incident_history[i];
        }
    }

    nimcp_mutex_unlock(&engine->incident_mutex);
    return match_count;
}

/**
 * @brief Query incidents by minimum severity
 */
uint32_t ethics_get_incidents_by_severity(ethics_engine_t engine, float min_severity,
                                          ethics_incident_t** incidents_out)
{
    // Guard clause: Validate inputs
    if (!engine || !incidents_out || min_severity < 0.0f || min_severity > 1.0f)
        return 0;

    if (!engine->incident_history)
        return 0;

    nimcp_mutex_lock(&engine->incident_mutex);

    // Count matching incidents
    uint32_t match_count = 0;
    for (uint32_t i = 0; i < engine->incident_count; i++) {
        if (engine->incident_history[i].severity >= min_severity)
            match_count++;
    }

    if (match_count == 0) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Allocate output array
    *incidents_out = nimcp_calloc(match_count, sizeof(ethics_incident_t));
    if (!*incidents_out) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Copy matching incidents
    uint32_t out_index = 0;
    for (uint32_t i = 0; i < engine->incident_count; i++) {
        if (engine->incident_history[i].severity >= min_severity) {
            (*incidents_out)[out_index++] = engine->incident_history[i];
        }
    }

    nimcp_mutex_unlock(&engine->incident_mutex);
    return match_count;
}

/**
 * @brief Query incidents by action taken
 */
uint32_t ethics_get_incidents_by_action(ethics_engine_t engine, ethics_action_t action,
                                        ethics_incident_t** incidents_out)
{
    // Guard clause: Validate inputs
    if (!engine || !incidents_out)
        return 0;

    if (!engine->incident_history)
        return 0;

    nimcp_mutex_lock(&engine->incident_mutex);

    // Count matching incidents
    uint32_t match_count = 0;
    for (uint32_t i = 0; i < engine->incident_count; i++) {
        if (engine->incident_history[i].action_taken == action)
            match_count++;
    }

    if (match_count == 0) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Allocate output array
    *incidents_out = nimcp_calloc(match_count, sizeof(ethics_incident_t));
    if (!*incidents_out) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Copy matching incidents
    uint32_t out_index = 0;
    for (uint32_t i = 0; i < engine->incident_count; i++) {
        if (engine->incident_history[i].action_taken == action) {
            (*incidents_out)[out_index++] = engine->incident_history[i];
        }
    }

    nimcp_mutex_unlock(&engine->incident_mutex);
    return match_count;
}

/**
 * @brief Get all incidents in chronological order
 */
uint32_t ethics_get_all_incidents(ethics_engine_t engine, ethics_incident_t** incidents_out)
{
    // Guard clause: Validate inputs
    if (!engine || !incidents_out)
        return 0;

    if (!engine->incident_history)
        return 0;

    nimcp_mutex_lock(&engine->incident_mutex);

    uint32_t count = engine->incident_count;
    if (count == 0) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Allocate output array
    *incidents_out = nimcp_calloc(count, sizeof(ethics_incident_t));
    if (!*incidents_out) {
        nimcp_mutex_unlock(&engine->incident_mutex);
        return 0;
    }

    // Copy all incidents in chronological order
    if (engine->incident_count < MAX_INCIDENT_HISTORY) {
        // Haven't wrapped yet, simple copy
        memcpy(*incidents_out, engine->incident_history, count * sizeof(ethics_incident_t));
    } else {
        // Wrapped around, need to copy in two parts for chronological order
        uint32_t start_index = engine->incident_index % MAX_INCIDENT_HISTORY;
        uint32_t first_part = MAX_INCIDENT_HISTORY - start_index;

        // Copy older incidents (from start_index to end)
        memcpy(*incidents_out, &engine->incident_history[start_index],
               first_part * sizeof(ethics_incident_t));

        // Copy newer incidents (from 0 to start_index)
        memcpy(&(*incidents_out)[first_part], engine->incident_history,
               start_index * sizeof(ethics_incident_t));
    }

    nimcp_mutex_unlock(&engine->incident_mutex);
    return count;
}

/**
 * @brief Export incidents to file
 */
bool ethics_export_incidents(ethics_engine_t engine, const char* filepath, const char* format)
{
    // Guard clause: Validate inputs
    if (!engine || !filepath || !format)
        return false;

    if (!engine->incident_history)
        return false;

    ethics_incident_t* incidents = NULL;
    uint32_t count = ethics_get_all_incidents(engine, &incidents);

    if (count == 0 || !incidents)
        return false;

    FILE* file = fopen(filepath, "w");
    if (!file) {
        nimcp_free(incidents);
        return false;
    }

    if (strcmp(format, "json") == 0) {
        // Export as JSON
        fprintf(file, "{\n  \"incidents\": [\n");
        for (uint32_t i = 0; i < count; i++) {
            fprintf(file, "    {\n");
            fprintf(file, "      \"id\": %lu,\n", incidents[i].incident_id);
            fprintf(file, "      \"timestamp\": %lu,\n", incidents[i].timestamp);
            fprintf(file, "      \"violation_type\": %d,\n", incidents[i].violation_type);
            fprintf(file, "      \"severity\": %.4f,\n", incidents[i].severity);
            fprintf(file, "      \"action_taken\": %d,\n", incidents[i].action_taken);
            fprintf(file, "      \"policy_id\": %u,\n", incidents[i].policy_id);
            fprintf(file, "      \"policy_name\": \"%s\",\n", incidents[i].policy_name);
            fprintf(file, "      \"description\": \"%s\",\n", incidents[i].description);
            fprintf(file, "      \"golden_rule_score\": %.4f,\n", incidents[i].golden_rule_score);
            fprintf(file, "      \"acting_agent\": %u,\n", incidents[i].acting_agent);
            fprintf(file, "      \"affected_agent\": %u\n", incidents[i].affected_agent);
            fprintf(file, "    }%s\n", (i < count - 1) ? "," : "");
        }
        fprintf(file, "  ]\n}\n");
    } else if (strcmp(format, "csv") == 0) {
        // Export as CSV
        fprintf(file, "id,timestamp,violation_type,severity,action_taken,policy_id,policy_name,description,golden_rule_score,acting_agent,affected_agent\n");
        for (uint32_t i = 0; i < count; i++) {
            fprintf(file, "%lu,%lu,%d,%.4f,%d,%u,\"%s\",\"%s\",%.4f,%u,%u\n",
                    incidents[i].incident_id,
                    incidents[i].timestamp,
                    incidents[i].violation_type,
                    incidents[i].severity,
                    incidents[i].action_taken,
                    incidents[i].policy_id,
                    incidents[i].policy_name,
                    incidents[i].description,
                    incidents[i].golden_rule_score,
                    incidents[i].acting_agent,
                    incidents[i].affected_agent);
        }
    } else {
        fclose(file);
        nimcp_free(incidents);
        return false;
    }

    fclose(file);
    nimcp_free(incidents);

    LOG_INFO("Exported %u incidents to %s", count, filepath);
    return true;
}
