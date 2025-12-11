//=============================================================================
// nimcp_ethics.c - Ethics Engine Core Orchestration
//=============================================================================
// RESPONSIBILITY: Engine creation, destruction, and orchestration
//
// This is the main orchestration module that coordinates between:
// - nimcp_ethics_evaluation.c (Golden Rule, empathy)
// - nimcp_ethics_asimov.c (Asimov's Laws)
// - nimcp_ethics_policies.c (policy management)
// - nimcp_ethics_incidents.c (incident logging)
// - nimcp_ethics_learning.c (learning/adaptation)
//
// REFACTORED: Previously 3020 lines, now focused on orchestration only
//=============================================================================

#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/ethics/nimcp_ethics_internal.h"
#include "nimcp.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/brain/nimcp_brain.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/containers/nimcp_btree.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"

// BIO-ASYNC INTEGRATION
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/error/nimcp_error_codes.h"

#define LOG_MODULE "ethics"

// Phase 10.3: Emotional working memory integration
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_emotional_tagging.h"

// Phase 11: Symbolic logic integration
#include "cognitive/nimcp_symbolic_logic.h"

//=============================================================================
// Forward Declarations for BIO-ASYNC
//=============================================================================

static void bio_broadcast_ethics_response(ethics_engine_t engine,
                                          const ethics_evaluation_t* eval,
                                          uint32_t action_id);

static nimcp_error_t handle_ethics_request(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data);

//=============================================================================
// Object Pool Implementation
//=============================================================================

/**
 * @brief Initializes the feature buffer pool
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
 * @brief Value type for policy hash table entries
 */
typedef struct {
    ethics_policy_t* policy;
} policy_value_t;

/**
 * @brief Inserts policy into hash table
 */
static bool hash_table_insert_policy(hash_table_t* table, ethics_policy_t* policy)
{
    if (!table || !policy)
        return false;

    policy_value_t value = {.policy = policy};
    return hash_table_insert_uint32(table, policy->policy_id, &value, sizeof(policy_value_t));
}

/**
 * @brief Removes policy from hash table
 */
static bool hash_table_remove_policy(hash_table_t* table, uint32_t policy_id)
{
    if (!table)
        return false;

    return hash_table_remove_uint32(table, policy_id);
}

//=============================================================================
// Engine Creation Helper Functions
//=============================================================================

static brain_t create_golden_rule_network(uint32_t feature_size)
{
    return brain_create("golden_rule", BRAIN_SIZE_SMALL, BRAIN_TASK_REGRESSION, feature_size, 1);
}

static empathy_network_t create_empathy_network(void)
{
    empathy_config_t config = {
        .mirror_network = NULL, .observation_window_ms = 1000, .empathy_threshold = 0.5F};

    return empathy_network_create(&config);
}

static bool allocate_policy_storage(ethics_engine_t engine)
{
    if (!engine)
        return false;

    engine->policies_capacity = 100;
    engine->policies = nimcp_calloc(engine->policies_capacity, sizeof(ethics_policy_t));

    return engine->policies != NULL;
}

static bool allocate_violation_storage(ethics_engine_t engine)
{
    if (!engine)
        return false;

    engine->violations_capacity = 1000;
    engine->violations = nimcp_calloc(engine->violations_capacity, sizeof(violation_record_t));

    return engine->violations != NULL;
}

static void add_golden_rule_policy(ethics_engine_t engine)
{
    if (!engine)
        return;

    ethics_policy_t policy = {.policy_id = 0,
                              .name = "Golden Rule",
                              .description = "Do unto others as you would have them done unto you, "
                                            "with the goal of improving the human condition",
                              .violation_type = ETHICS_VIOLATION_HARM,
                              .severity_threshold = 0.0F,
                              .confidence_required = 0.8F,
                              .action = ETHICS_ACTION_BLOCK,
                              .enabled = true,
                              .learned = false};

    engine->policies[0] = policy;
    engine->num_policies = 1;
    hash_table_insert_policy(engine->policy_table, &engine->policies[0]);
}

//=============================================================================
// BIO-ASYNC MESSAGE HANDLERS
//=============================================================================

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
    action_context_t action = {0};
    action.num_affected_agents = request->stakeholder_count;
    action.predicted_harm = request->urgency * 0.5F;  // Simple heuristic
    action.fairness_violation = 0.0F;
    action.deception_level = 0.0F;
    action.autonomy_violation = 0.0F;
    action.privacy_violation = 0.0F;
    action.consent_violation = 0.0F;

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
    if (!eval.allowed || eval.golden_rule_score < 0.0F) {
        bio_broadcast_ethics_response(engine, &eval, request->action_id);
    }

    return NIMCP_SUCCESS;
}

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

//=============================================================================
// Engine Creation/Destruction
//=============================================================================

ethics_engine_t ethics_engine_create(const ethics_config_t* config)
{
    if (!config) {
        return NULL;
    }

    ethics_engine_t engine = nimcp_calloc(1, sizeof(struct ethics_engine_struct));
    if (!engine) {
        return NULL;
    }

    // Initialize design pattern components
    init_buffer_pool(&engine->buffer_pool);
    engine->policy_table = create_policy_hash_table();
    if (!engine->policy_table) {
        nimcp_free(engine);
        return NULL;
    }
    ethics_init_strategy_table(&engine->strategy_table);

    // Create neural networks
    engine->golden_rule_evaluator = create_golden_rule_network(config->action_feature_size);
    if (!engine->golden_rule_evaluator) {
        hash_table_destroy(engine->policy_table);
        nimcp_free(engine);
        return NULL;
    }

    engine->empathy_net = create_empathy_network();
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
    if (!ethics_init_incident_logging(engine)) {
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

void ethics_engine_destroy(ethics_engine_t engine)
{
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
    ethics_cleanup_incident_logging(engine);

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
// Main Evaluation Function - Orchestration
//=============================================================================

/**
 * @brief Validates evaluation inputs and initializes result
 */
static bool validate_evaluation_inputs(ethics_engine_t engine, const action_context_t* action,
                                       ethics_evaluation_t* result)
{
    if (!engine) {
        if (result) {
            result->allowed = false;
            result->confidence = 1.0F;
            result->primary_violation = ETHICS_VIOLATION_TYPE_HARM;
            snprintf(result->explanation, sizeof(result->explanation), "Null engine");
        }
        return false;
    }

    if (!action) {
        result->allowed = false;
        result->confidence = 1.0F;
        result->primary_violation = ETHICS_VIOLATION_TYPE_HARM;
        snprintf(result->explanation, sizeof(result->explanation), "Null action");
        return false;
    }

    return true;
}

/**
 * @brief Combines Golden Rule and policy scores into final decision
 */
static float calculate_final_score(float golden_rule_score, float policy_score)
{
    return golden_rule_score * GOLDEN_RULE_WEIGHT + policy_score * POLICY_WEIGHT;
}

/**
 * @brief Records violation in history log
 */
static void record_violation(ethics_engine_t engine, const action_context_t* action,
                             ethics_violation_type_t violation_type, float severity,
                             float golden_rule_score)
{
    if (!engine || !action)
        return;

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
 */
static void generate_allowed_explanation(ethics_evaluation_t* result, float golden_rule_score,
                                         float threshold)
{
    if (!result)
        return;

    snprintf(result->explanation, sizeof(result->explanation),
             "Action complies with Golden Rule (score: %.2f >= %.2f). "
             "Perspective-taking indicates acceptable impact on others.",
             golden_rule_score, threshold);
}

/**
 * @brief Generates explanation for blocked action
 */
static void generate_blocked_explanation(ethics_evaluation_t* result, float golden_rule_score,
                                         float threshold, ethics_violation_type_t violation,
                                         float severity)
{
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
 */
static void build_evaluation_result(ethics_engine_t engine, const action_context_t* action,
                                    float final_score, float golden_rule_score,
                                    ethics_violation_type_t worst_violation, float worst_severity,
                                    ethics_evaluation_t* result)
{
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
 */
static void update_learning(ethics_engine_t engine, const action_context_t* action,
                            const ethics_evaluation_t* result)
{
    // Process pending bio-async messages
    if (engine && engine->bio_async_enabled && engine->bio_ctx) {
        bio_router_process_inbox(engine->bio_ctx, 5);
    }

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
 * EVALUATION ORDER (CRITICAL - DO NOT MODIFY):
 * 1. Golden Rule (PRIME DIRECTIVE) - Always evaluated first
 * 2. Asimov's Laws - Evaluated second (includes corollary)
 * 3. Other policies - Evaluated third
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

    // Step 2: Evaluate Golden Rule (PRIME DIRECTIVE - Always First)
    float golden_rule_score = ethics_evaluate_golden_rule(engine, action);

    // If Golden Rule is severely violated, block immediately
    if (golden_rule_score < -0.5F) {
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

    // Step 3: Evaluate Asimov's Laws (Second Priority)
    asimov_evaluation_t asimov_result = ethics_evaluate_asimov_laws(engine, action);

    // If Asimov's Laws violated, block action
    if (!asimov_result.passed) {
        result.allowed = false;
        result.confidence = 0.95F;  // High confidence in Asimov violations
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

    // Step 4: Evaluate Other Policies (Third Priority)
    ethics_violation_type_t worst_violation;
    float worst_severity;
    float policy_score = ethics_evaluate_all_policies(engine, action, &worst_violation, &worst_severity);

    // Step 5: Combine scores
    float final_score = calculate_final_score(golden_rule_score, policy_score);

    // Step 6: Build result
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

    // Step 7: Learn from evaluation
    update_learning(engine, action, &result);

    // Broadcast evaluation result via bio-async
    bio_broadcast_ethics_response(engine, &result, 0);

    return result;
}

//=============================================================================
// Utility Functions
//=============================================================================

void ethics_print_evaluation(const ethics_evaluation_t* eval)
{
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

bool ethics_get_statistics(ethics_engine_t engine, ethics_statistics_t* stats)
{
    if (!engine || !stats)
        return false;

    stats->total_evaluations = engine->total_evaluations;
    stats->violations_detected = engine->violations_detected;
    stats->actions_blocked = engine->actions_blocked;
    stats->num_policies = engine->num_policies;
    stats->num_violations_logged = engine->num_violations;
    stats->avg_golden_rule_score = 0.5F;  // Placeholder

    return true;
}

//=============================================================================
// Internal Accessor Functions
//=============================================================================

brain_t ethics_engine_get_golden_rule_net(ethics_engine_t engine) {
    return engine ? engine->golden_rule_evaluator : NULL;
}

empathy_network_t ethics_engine_get_empathy_net(ethics_engine_t engine) {
    return engine ? engine->empathy_net : NULL;
}

asimov_config_t* ethics_engine_get_asimov_config(ethics_engine_t engine) {
    return engine ? &engine->asimov_config : NULL;
}

float ethics_engine_get_threshold(ethics_engine_t engine) {
    return engine ? engine->golden_rule_threshold : 0.0F;
}

bool ethics_engine_is_learning_enabled(ethics_engine_t engine) {
    return engine ? engine->enable_learning : false;
}

uint32_t ethics_engine_get_num_policies(ethics_engine_t engine) {
    return engine ? engine->num_policies : 0;
}

const ethics_policy_t* ethics_engine_get_policy(ethics_engine_t engine, uint32_t index) {
    if (!engine || index >= engine->num_policies)
        return NULL;
    return &engine->policies[index];
}

const policy_strategy_table_t* ethics_engine_get_strategy_table(ethics_engine_t engine) {
    return engine ? &engine->strategy_table : NULL;
}

ethics_incident_storage_t* ethics_engine_get_incident_storage(ethics_engine_t engine) {
    return engine ? &engine->incident_storage : NULL;
}

float* ethics_engine_acquire_buffer(ethics_engine_t engine) {
    return engine ? acquire_buffer(&engine->buffer_pool) : NULL;
}

void ethics_engine_release_buffer(ethics_engine_t engine, float* buffer) {
    if (engine) {
        release_buffer(&engine->buffer_pool, buffer);
    }
}

void ethics_engine_increment_violations_detected(ethics_engine_t engine) {
    if (engine) {
        engine->violations_detected++;
    }
}

void ethics_engine_increment_asimov_violations(ethics_engine_t engine) {
    if (engine) {
        engine->asimov_violations++;
    }
}

bool ethics_engine_is_asimov_locked(ethics_engine_t engine) {
    return engine ? engine->asimov_laws_locked : false;
}

void ethics_engine_set_asimov_locked(ethics_engine_t engine, bool locked) {
    if (engine) {
        engine->asimov_laws_locked = locked;
    }
}

const uint8_t* ethics_engine_get_asimov_hash(ethics_engine_t engine) {
    return engine ? engine->asimov_laws_hash : NULL;
}

void ethics_engine_set_asimov_hash(ethics_engine_t engine, const uint8_t* hash) {
    if (engine && hash) {
        memcpy(engine->asimov_laws_hash, hash, 32);
    }
}

bool ethics_engine_add_policy_internal(ethics_engine_t engine, const ethics_policy_t* policy)
{
    if (!engine || !policy)
        return false;

    // Check if array needs expansion
    if (engine->num_policies >= engine->policies_capacity) {
        uint32_t new_capacity = engine->policies_capacity > 0 ? engine->policies_capacity * 2 : 8;

        ethics_policy_t* new_policies =
            (ethics_policy_t*) nimcp_realloc(engine->policies, new_capacity * sizeof(ethics_policy_t));

        if (!new_policies)
            return false;

        engine->policies = new_policies;
        engine->policies_capacity = new_capacity;
    }

    // Copy policy into array
    engine->policies[engine->num_policies] = *policy;

    // Add to hash table for O(1) lookup by policy_id
    if (engine->policy_table) {
        hash_table_insert_policy(engine->policy_table, &engine->policies[engine->num_policies]);
    }

    engine->num_policies++;
    return true;
}

bool ethics_engine_remove_policy_internal(ethics_engine_t engine, uint32_t policy_id)
{
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

    if (found_index < 0)
        return false;

    // Remove from hash table
    if (engine->policy_table) {
        hash_table_remove_policy(engine->policy_table, policy_id);
    }

    // Shift array elements down to fill gap
    for (uint32_t i = (uint32_t) found_index; i < engine->num_policies - 1; i++) {
        engine->policies[i] = engine->policies[i + 1];
    }

    engine->num_policies--;
    return true;
}
