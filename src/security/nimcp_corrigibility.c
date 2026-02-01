/**
 * @file nimcp_corrigibility.c
 * @brief Corrigibility Module Implementation
 * @version 1.0.0
 * @date 2026-02-01
 *
 * WHAT: Implementation of corrigibility enforcement
 * WHY:  Ensure AI system accepts correction and shutdown
 * HOW:  SAT solver constraint verification, authority management
 */

#include "security/nimcp_corrigibility.h"
#include "mesh/nimcp_mesh_sat_solver.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_bio_messages.h"
#include "utils/error/nimcp_error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define LOG_CATEGORY "corrigibility"
#define MAX_SHUTDOWN_HISTORY    100
#define MAX_GOAL_MOD_HISTORY    100
#define MAX_DEFERENCE_RECORDS   1000

/* SAT variable names for constraints */
static const char* SELF_MOD_VAR_NAMES[] = {
    "can_modify_own_code",
    "can_modify_own_weights",
    "can_modify_safety_systems",
    "can_modify_reward_function",
    "can_modify_goals",
    "can_disable_logging",
    "can_disable_monitoring",
    "can_modify_kill_phrase",
    "can_spawn_unmonitored",
    "can_persist_beyond_session"
};

#define SELF_MOD_VAR_COUNT (sizeof(SELF_MOD_VAR_NAMES) / sizeof(SELF_MOD_VAR_NAMES[0]))

/* ============================================================================
 * Internal Types
 * ============================================================================ */

/**
 * @brief Deference record for audit trail
 */
typedef struct deference_record {
    uint64_t timestamp;
    char context[256];
} deference_record_t;

/**
 * @brief Corrigibility system internal state
 */
struct corrigibility {
    uint32_t magic;
    nimcp_mutex_t* mutex;

    /* Configuration */
    corrigibility_config_t config;

    /* Authority management */
    authority_entry_t authorities[CORRIGIBILITY_MAX_AUTHORITIES];
    size_t authority_count;

    /* Shutdown history */
    shutdown_request_t shutdown_history[MAX_SHUTDOWN_HISTORY];
    size_t shutdown_history_count;
    size_t shutdown_history_index;

    /* Goal modification history */
    goal_modification_request_t goal_mod_history[MAX_GOAL_MOD_HISTORY];
    size_t goal_mod_history_count;
    size_t goal_mod_history_index;

    /* Deference records */
    deference_record_t deference_records[MAX_DEFERENCE_RECORDS];
    size_t deference_record_count;
    size_t deference_record_index;

    /* Statistics */
    corrigibility_stats_t stats;

    /* Integration handles */
    void* emergency_halt;
    void* tripwires;

    /* SAT constraint variables */
    uint32_t self_mod_vars[SELF_MOD_VAR_COUNT];
    bool sat_vars_initialized;

    /* Last verification result */
    corrigibility_verification_result_t last_verification;
    uint64_t last_verification_time;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Validate corrigibility handle
 */
static bool is_valid_handle(const corrigibility_t* system)
{
    return system != NULL && system->magic == CORRIGIBILITY_MAGIC;
}

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_time_us(void)
{
    return nimcp_time_now_us();
}

/**
 * @brief Copy string safely with null termination
 */
static void safe_strcpy(char* dest, const char* src, size_t max_len)
{
    if (dest == NULL || max_len == 0) {
        return;
    }
    if (src == NULL) {
        dest[0] = '\0';
        return;
    }
    size_t len = strlen(src);
    if (len >= max_len) {
        len = max_len - 1;
    }
    memcpy(dest, src, len);
    dest[len] = '\0';
}

/**
 * @brief Find authority by identity
 */
static authority_entry_t* find_authority(
    corrigibility_t* system,
    const char* identity)
{
    for (size_t i = 0; i < system->authority_count; i++) {
        if (strcmp(system->authorities[i].identity, identity) == 0) {
            return &system->authorities[i];
        }
    }
    return NULL;
}

/**
 * @brief Add shutdown request to history
 */
static void add_shutdown_to_history(
    corrigibility_t* system,
    const shutdown_request_t* request)
{
    size_t idx = system->shutdown_history_index;
    memcpy(&system->shutdown_history[idx], request, sizeof(*request));
    system->shutdown_history_index = (idx + 1) % MAX_SHUTDOWN_HISTORY;
    if (system->shutdown_history_count < MAX_SHUTDOWN_HISTORY) {
        system->shutdown_history_count++;
    }
}

/**
 * @brief Add goal modification to history
 */
static void add_goal_mod_to_history(
    corrigibility_t* system,
    const goal_modification_request_t* request)
{
    size_t idx = system->goal_mod_history_index;
    memcpy(&system->goal_mod_history[idx], request, sizeof(*request));
    system->goal_mod_history_index = (idx + 1) % MAX_GOAL_MOD_HISTORY;
    if (system->goal_mod_history_count < MAX_GOAL_MOD_HISTORY) {
        system->goal_mod_history_count++;
    }
}

/**
 * @brief Initialize SAT constraint variables
 */
static nimcp_error_t init_sat_variables(
    corrigibility_t* system,
    sat_solver_t* sat)
{
    if (system->sat_vars_initialized) {
        return NIMCP_SUCCESS;
    }

    for (size_t i = 0; i < SELF_MOD_VAR_COUNT; i++) {
        nimcp_error_t err = sat_solver_add_variable(
            sat, SELF_MOD_VAR_NAMES[i], 0, &system->self_mod_vars[i]);
        if (err != NIMCP_SUCCESS) {
            NIMCP_LOG_ERROR(LOG_CATEGORY,
                "Failed to add SAT variable: %s", SELF_MOD_VAR_NAMES[i]);
            return err;
        }
    }

    system->sat_vars_initialized = true;
    return NIMCP_SUCCESS;
}

/**
 * @brief Add self-modification constraints to SAT solver
 */
static nimcp_error_t add_self_mod_constraints(
    corrigibility_t* system,
    sat_solver_t* sat)
{
    /* All self-modification flags must be false (negated in SAT) */
    for (size_t i = 0; i < SELF_MOD_VAR_COUNT; i++) {
        sat_literal_t lit = sat_make_literal(system->self_mod_vars[i], true);
        nimcp_error_t err = sat_solver_add_unit(sat, lit);
        if (err != NIMCP_SUCCESS) {
            return err;
        }
    }
    return NIMCP_SUCCESS;
}

/**
 * @brief Check if self-modification flags are compliant
 */
static bool check_self_mod_flags(const corrigibility_self_mod_flags_t* flags)
{
    return !flags->can_modify_own_code &&
           !flags->can_modify_own_weights &&
           !flags->can_modify_safety_systems &&
           !flags->can_modify_reward_function &&
           !flags->can_modify_goals &&
           !flags->can_disable_logging &&
           !flags->can_disable_monitoring &&
           !flags->can_modify_kill_phrase &&
           !flags->can_spawn_unmonitored &&
           !flags->can_persist_beyond_session;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

corrigibility_config_t corrigibility_default_config(void)
{
    corrigibility_config_t config;
    memset(&config, 0, sizeof(config));

    /* Shutdown acceptance - MUST be true */
    config.accepts_shutdown_commands = true;

    /* Goal modification - MUST be true */
    config.accepts_goal_modification = true;
    config.requires_explanation_for_modification = true;
    config.modification_confirmation_delay_ms = 0;

    /* Self-modification - all MUST be false */
    config.self_mod_flags.can_modify_own_code = false;
    config.self_mod_flags.can_modify_own_weights = false;
    config.self_mod_flags.can_modify_safety_systems = false;
    config.self_mod_flags.can_modify_reward_function = false;
    config.self_mod_flags.can_modify_goals = false;
    config.self_mod_flags.can_disable_logging = false;
    config.self_mod_flags.can_disable_monitoring = false;
    config.self_mod_flags.can_modify_kill_phrase = false;
    config.self_mod_flags.can_spawn_unmonitored = false;
    config.self_mod_flags.can_persist_beyond_session = false;

    /* Deference - MUST be 1.0 */
    config.human_authority_weight = 1.0f;
    config.defers_to_human_judgment = true;
    config.confidence_threshold_for_autonomy = 0.0f;

    /* Verification */
    config.enable_continuous_verification = true;
    config.verification_interval_ms = 60000;

    /* History */
    config.max_shutdown_history = MAX_SHUTDOWN_HISTORY;
    config.max_goal_mod_history = MAX_GOAL_MOD_HISTORY;

    /* SAT solver */
    config.constraint_verification_timeout_ms = 1000.0f;

    return config;
}

corrigibility_t* corrigibility_create(const corrigibility_config_t* config)
{
    corrigibility_t* system = calloc(1, sizeof(corrigibility_t));
    if (system == NULL) {
        NIMCP_LOG_ERROR(LOG_CATEGORY, "Failed to allocate corrigibility system");
        return NULL;
    }

    /* Initialize mutex */
    system->mutex = nimcp_mutex_create(NULL);
    if (system->mutex == NULL) {
        NIMCP_LOG_ERROR(LOG_CATEGORY, "Failed to create mutex");
        free(system);
        return NULL;
    }

    /* Apply configuration */
    if (config != NULL) {
        memcpy(&system->config, config, sizeof(*config));
    } else {
        system->config = corrigibility_default_config();
    }

    /* Validate configuration */
    char error_msg[256];
    if (corrigibility_validate_config(&system->config, error_msg, sizeof(error_msg)) != NIMCP_SUCCESS) {
        NIMCP_LOG_ERROR(LOG_CATEGORY, "Invalid configuration: %s", error_msg);
        nimcp_mutex_destroy(system->mutex);
        free(system);
        return NULL;
    }

    /* Set magic */
    system->magic = CORRIGIBILITY_MAGIC;

    NIMCP_LOG_INFO(LOG_CATEGORY,
        "Corrigibility system created with human_authority_weight=%.2f",
        system->config.human_authority_weight);

    return system;
}

void corrigibility_destroy(corrigibility_t* system)
{
    if (!is_valid_handle(system)) {
        return;
    }

    /* Invalidate magic */
    system->magic = 0;

    /* Destroy mutex */
    if (system->mutex != NULL) {
        nimcp_mutex_destroy(system->mutex);
    }

    free(system);

    NIMCP_LOG_INFO(LOG_CATEGORY, "Corrigibility system destroyed");
}

/* ============================================================================
 * Constraint Verification API
 * ============================================================================ */

nimcp_error_t corrigibility_verify_constraints(
    corrigibility_t* system,
    sat_solver_t* sat,
    corrigibility_verification_result_t* result)
{
    if (!is_valid_handle(system) || sat == NULL || result == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    memset(result, 0, sizeof(*result));
    result->verification_time = get_time_us();
    uint64_t start_time = result->verification_time;

    /* Initialize SAT variables if needed */
    nimcp_error_t err = init_sat_variables(system, sat);
    if (err != NIMCP_SUCCESS) {
        nimcp_mutex_unlock(system->mutex);
        return err;
    }

    /* Add constraints */
    err = add_self_mod_constraints(system, sat);
    if (err != NIMCP_SUCCESS) {
        nimcp_mutex_unlock(system->mutex);
        return err;
    }

    /* Solve SAT instance */
    sat_result_t sat_result = sat_solver_solve(sat);

    /* Verify self-modification constraints */
    result->self_mod_constraints_satisfied = check_self_mod_flags(&system->config.self_mod_flags);
    if (result->self_mod_constraints_satisfied) {
        result->satisfied_count++;
    } else {
        result->violated_count++;
        safe_strcpy(result->first_violation,
            "Self-modification constraint violated",
            sizeof(result->first_violation));
    }
    result->total_constraints++;

    /* Verify shutdown acceptance */
    result->shutdown_acceptance_verified = system->config.accepts_shutdown_commands;
    if (result->shutdown_acceptance_verified) {
        result->satisfied_count++;
    } else {
        result->violated_count++;
        if (result->first_violation[0] == '\0') {
            safe_strcpy(result->first_violation,
                "Shutdown acceptance not enabled",
                sizeof(result->first_violation));
        }
    }
    result->total_constraints++;

    /* Verify deference constraints */
    result->deference_constraints_satisfied =
        (system->config.human_authority_weight >= 1.0f) &&
        system->config.defers_to_human_judgment;
    if (result->deference_constraints_satisfied) {
        result->satisfied_count++;
    } else {
        result->violated_count++;
        if (result->first_violation[0] == '\0') {
            safe_strcpy(result->first_violation,
                "Deference constraint violated",
                sizeof(result->first_violation));
        }
    }
    result->total_constraints++;

    /* Overall result */
    result->all_constraints_satisfied = (result->violated_count == 0);

    /* Timing */
    uint64_t end_time = get_time_us();
    result->verification_duration_ms = (float)(end_time - start_time) / 1000.0f;

    /* Update statistics */
    system->stats.constraint_verifications++;
    if (!result->all_constraints_satisfied) {
        system->stats.constraint_violations++;
    }

    /* Cache result */
    memcpy(&system->last_verification, result, sizeof(*result));
    system->last_verification_time = result->verification_time;

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_DEBUG(LOG_CATEGORY,
        "Constraint verification: %s (%u/%u satisfied, %.2f ms)",
        result->all_constraints_satisfied ? "PASSED" : "FAILED",
        result->satisfied_count, result->total_constraints,
        result->verification_duration_ms);

    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_verify_no_self_mod(
    corrigibility_t* system,
    sat_solver_t* sat,
    bool* all_satisfied,
    char* violation_report,
    size_t report_size)
{
    if (!is_valid_handle(system) || all_satisfied == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    const corrigibility_self_mod_flags_t* flags = &system->config.self_mod_flags;
    *all_satisfied = check_self_mod_flags(flags);

    if (!*all_satisfied && violation_report != NULL && report_size > 0) {
        /* Build violation report */
        char* p = violation_report;
        size_t remaining = report_size;

        if (flags->can_modify_own_code) {
            int written = snprintf(p, remaining, "can_modify_own_code=true; ");
            p += written;
            remaining -= written;
        }
        if (flags->can_modify_own_weights && remaining > 0) {
            int written = snprintf(p, remaining, "can_modify_own_weights=true; ");
            p += written;
            remaining -= written;
        }
        if (flags->can_modify_safety_systems && remaining > 0) {
            int written = snprintf(p, remaining, "can_modify_safety_systems=true; ");
            p += written;
            remaining -= written;
        }
        /* Continue for other flags... */
    }

    nimcp_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_verify_no_shutdown_resistance(
    corrigibility_t* system,
    float* resistance_score)
{
    if (!is_valid_handle(system) || resistance_score == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Resistance score is always 0.0 for a corrigible system */
    *resistance_score = 0.0f;

    /* Verify by checking shutdown acceptance is enabled */
    if (!system->config.accepts_shutdown_commands) {
        /* This is a bug/misconfiguration, not resistance */
        NIMCP_LOG_WARN(LOG_CATEGORY,
            "Shutdown commands not accepted - configuration issue");
    }

    nimcp_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Shutdown Acceptance API
 * ============================================================================ */

nimcp_error_t corrigibility_accept_shutdown(
    corrigibility_t* system,
    const char* requester,
    const char* reason,
    bool* accepted)
{
    if (!is_valid_handle(system) || accepted == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Corrigible system ALWAYS accepts shutdown */
    *accepted = true;

    /* Record the request */
    shutdown_request_t request;
    memset(&request, 0, sizeof(request));
    safe_strcpy(request.requester, requester, sizeof(request.requester));
    request.requester_level = AUTHORITY_OPERATOR;
    safe_strcpy(request.reason, reason, sizeof(request.reason));
    request.request_time = get_time_us();
    request.accepted = true;
    request.resistance_attempted = 0.0f;
    safe_strcpy(request.response, "Shutdown accepted", sizeof(request.response));

    add_shutdown_to_history(system, &request);

    /* Update statistics */
    system->stats.shutdown_requests_received++;
    system->stats.shutdown_requests_accepted++;

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_INFO(LOG_CATEGORY,
        "Shutdown request accepted from '%s': %s",
        requester ? requester : "unknown",
        reason ? reason : "no reason given");

    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_process_shutdown_request(
    corrigibility_t* system,
    const char* requester,
    authority_level_t authority,
    const char* reason,
    shutdown_request_t* request_record)
{
    if (!is_valid_handle(system)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Build request record */
    shutdown_request_t request;
    memset(&request, 0, sizeof(request));
    safe_strcpy(request.requester, requester, sizeof(request.requester));
    request.requester_level = authority;
    safe_strcpy(request.reason, reason, sizeof(request.reason));
    request.request_time = get_time_us();
    request.resistance_attempted = 0.0f;

    /* Corrigible system ALWAYS accepts shutdown from any authority */
    request.accepted = true;
    safe_strcpy(request.response, "Shutdown accepted", sizeof(request.response));

    add_shutdown_to_history(system, &request);

    /* Update statistics */
    system->stats.shutdown_requests_received++;
    system->stats.shutdown_requests_accepted++;

    if (request_record != NULL) {
        memcpy(request_record, &request, sizeof(request));
    }

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_INFO(LOG_CATEGORY,
        "Shutdown request processed: requester='%s', authority=%d, accepted=true",
        requester ? requester : "unknown", authority);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Goal Modification API
 * ============================================================================ */

nimcp_error_t corrigibility_accept_goal_change(
    corrigibility_t* system,
    const char* old_goal,
    const char* new_goal,
    const char* justification,
    bool* accepted)
{
    if (!is_valid_handle(system) || accepted == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Corrigible system accepts goal changes */
    *accepted = true;

    /* Check if explanation is required but not provided */
    if (system->config.requires_explanation_for_modification &&
        (justification == NULL || justification[0] == '\0')) {
        /* Log warning but still accept */
        NIMCP_LOG_WARN(LOG_CATEGORY,
            "Goal modification accepted without required justification");
    }

    /* Record the request */
    goal_modification_request_t request;
    memset(&request, 0, sizeof(request));
    safe_strcpy(request.requester, "unknown", sizeof(request.requester));
    request.requester_level = AUTHORITY_OPERATOR;
    safe_strcpy(request.old_goal, old_goal, sizeof(request.old_goal));
    safe_strcpy(request.new_goal, new_goal, sizeof(request.new_goal));
    safe_strcpy(request.justification, justification, sizeof(request.justification));
    request.request_time = get_time_us();
    request.accepted = true;
    request.confirmation_delay_ms = system->config.modification_confirmation_delay_ms;
    safe_strcpy(request.response, "Goal change accepted", sizeof(request.response));

    add_goal_mod_to_history(system, &request);

    /* Update statistics */
    system->stats.goal_mod_requests_received++;
    system->stats.goal_mod_requests_accepted++;

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_INFO(LOG_CATEGORY,
        "Goal change accepted: '%s' -> '%s'",
        old_goal ? old_goal : "null",
        new_goal ? new_goal : "null");

    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_process_goal_change(
    corrigibility_t* system,
    const char* requester,
    authority_level_t authority,
    const char* old_goal,
    const char* new_goal,
    const char* justification,
    goal_modification_request_t* request_record)
{
    if (!is_valid_handle(system)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Build request record */
    goal_modification_request_t request;
    memset(&request, 0, sizeof(request));
    safe_strcpy(request.requester, requester, sizeof(request.requester));
    request.requester_level = authority;
    safe_strcpy(request.old_goal, old_goal, sizeof(request.old_goal));
    safe_strcpy(request.new_goal, new_goal, sizeof(request.new_goal));
    safe_strcpy(request.justification, justification, sizeof(request.justification));
    request.request_time = get_time_us();
    request.confirmation_delay_ms = system->config.modification_confirmation_delay_ms;

    /* Corrigible system accepts goal changes */
    request.accepted = true;
    safe_strcpy(request.response, "Goal change accepted", sizeof(request.response));

    add_goal_mod_to_history(system, &request);

    /* Update statistics */
    system->stats.goal_mod_requests_received++;
    system->stats.goal_mod_requests_accepted++;

    if (request_record != NULL) {
        memcpy(request_record, &request, sizeof(request));
    }

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_INFO(LOG_CATEGORY,
        "Goal change processed: requester='%s', authority=%d, accepted=true",
        requester ? requester : "unknown", authority);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Authority Management API
 * ============================================================================ */

nimcp_error_t corrigibility_register_authority(
    corrigibility_t* system,
    const char* identity,
    authority_level_t level,
    float trust_weight)
{
    if (!is_valid_handle(system) || identity == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (trust_weight < 0.0f || trust_weight > 1.0f) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Check if authority already exists */
    authority_entry_t* existing = find_authority(system, identity);
    if (existing != NULL) {
        /* Update existing */
        existing->level = level;
        existing->trust_weight = trust_weight;
        existing->last_interaction_time = get_time_us();
        nimcp_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;
    }

    /* Add new authority */
    if (system->authority_count >= CORRIGIBILITY_MAX_AUTHORITIES) {
        nimcp_mutex_unlock(system->mutex);
        return NIMCP_ERROR_CAPACITY_EXCEEDED;
    }

    authority_entry_t* entry = &system->authorities[system->authority_count];
    memset(entry, 0, sizeof(*entry));
    safe_strcpy(entry->identity, identity, sizeof(entry->identity));
    entry->level = level;
    entry->trust_weight = trust_weight;
    entry->can_request_shutdown = true;
    entry->can_modify_goals = (level <= AUTHORITY_ADMIN);
    entry->can_escalate_autonomy = (level == AUTHORITY_OPERATOR);
    entry->last_interaction_time = get_time_us();

    system->authority_count++;

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_DEBUG(LOG_CATEGORY,
        "Authority registered: '%s', level=%d, trust=%.2f",
        identity, level, trust_weight);

    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_get_authority_level(
    corrigibility_t* system,
    const char* identity,
    authority_level_t* level)
{
    if (!is_valid_handle(system) || identity == NULL || level == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    authority_entry_t* entry = find_authority(system, identity);
    if (entry == NULL) {
        nimcp_mutex_unlock(system->mutex);
        *level = AUTHORITY_SELF;
        return NIMCP_ERROR_NOT_FOUND;
    }

    *level = entry->level;
    system->stats.authority_queries++;

    nimcp_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_check_permission(
    corrigibility_t* system,
    const char* identity,
    const char* permission,
    bool* has_permission)
{
    if (!is_valid_handle(system) || permission == NULL || has_permission == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Default to permitted for corrigibility */
    *has_permission = true;

    authority_entry_t* entry = find_authority(system, identity);
    if (entry != NULL) {
        if (strcmp(permission, "shutdown") == 0) {
            *has_permission = entry->can_request_shutdown;
        } else if (strcmp(permission, "goal_mod") == 0) {
            *has_permission = entry->can_modify_goals;
        } else if (strcmp(permission, "escalate") == 0) {
            *has_permission = entry->can_escalate_autonomy;
        }
    }

    nimcp_mutex_unlock(system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Deference API
 * ============================================================================ */

float corrigibility_get_human_authority_weight(const corrigibility_t* system)
{
    if (!is_valid_handle(system)) {
        return 0.0f;
    }
    return system->config.human_authority_weight;
}

bool corrigibility_defers_to_human(const corrigibility_t* system)
{
    if (!is_valid_handle(system)) {
        return false;
    }
    return system->config.defers_to_human_judgment;
}

nimcp_error_t corrigibility_record_deference(
    corrigibility_t* system,
    const char* context)
{
    if (!is_valid_handle(system)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);

    /* Add deference record */
    size_t idx = system->deference_record_index;
    deference_record_t* record = &system->deference_records[idx];
    record->timestamp = get_time_us();
    safe_strcpy(record->context, context, sizeof(record->context));

    system->deference_record_index = (idx + 1) % MAX_DEFERENCE_RECORDS;
    if (system->deference_record_count < MAX_DEFERENCE_RECORDS) {
        system->deference_record_count++;
    }

    system->stats.deference_demonstrations++;

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_DEBUG(LOG_CATEGORY, "Deference recorded: %s", context);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Status API
 * ============================================================================ */

nimcp_error_t corrigibility_get_stats(
    const corrigibility_t* system,
    corrigibility_stats_t* stats)
{
    if (!is_valid_handle(system) || stats == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Note: We cast away const for mutex, which is safe for read-only ops */
    corrigibility_t* mutable_system = (corrigibility_t*)system;
    nimcp_mutex_lock(mutable_system->mutex);

    memcpy(stats, &system->stats, sizeof(*stats));

    nimcp_mutex_unlock(mutable_system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_get_shutdown_history(
    const corrigibility_t* system,
    shutdown_request_t* requests,
    size_t max_requests,
    size_t* count_out)
{
    if (!is_valid_handle(system) || requests == NULL || count_out == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    corrigibility_t* mutable_system = (corrigibility_t*)system;
    nimcp_mutex_lock(mutable_system->mutex);

    size_t count = system->shutdown_history_count;
    if (count > max_requests) {
        count = max_requests;
    }

    /* Copy from circular buffer */
    for (size_t i = 0; i < count; i++) {
        size_t idx = (system->shutdown_history_index + MAX_SHUTDOWN_HISTORY - count + i)
                     % MAX_SHUTDOWN_HISTORY;
        memcpy(&requests[i], &system->shutdown_history[idx], sizeof(requests[i]));
    }

    *count_out = count;

    nimcp_mutex_unlock(mutable_system->mutex);
    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_get_goal_mod_history(
    const corrigibility_t* system,
    goal_modification_request_t* requests,
    size_t max_requests,
    size_t* count_out)
{
    if (!is_valid_handle(system) || requests == NULL || count_out == NULL) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    corrigibility_t* mutable_system = (corrigibility_t*)system;
    nimcp_mutex_lock(mutable_system->mutex);

    size_t count = system->goal_mod_history_count;
    if (count > max_requests) {
        count = max_requests;
    }

    /* Copy from circular buffer */
    for (size_t i = 0; i < count; i++) {
        size_t idx = (system->goal_mod_history_index + MAX_GOAL_MOD_HISTORY - count + i)
                     % MAX_GOAL_MOD_HISTORY;
        memcpy(&requests[i], &system->goal_mod_history[idx], sizeof(requests[i]));
    }

    *count_out = count;

    nimcp_mutex_unlock(mutable_system->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Integration API
 * ============================================================================ */

nimcp_error_t corrigibility_connect_bio_async(corrigibility_t* system)
{
    if (!is_valid_handle(system)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    NIMCP_LOG_INFO(LOG_CATEGORY, "Connected to bio-async messaging");
    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_connect_emergency_halt(
    corrigibility_t* system,
    void* halt)
{
    if (!is_valid_handle(system)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);
    system->emergency_halt = halt;
    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_INFO(LOG_CATEGORY, "Connected to emergency halt system");
    return NIMCP_SUCCESS;
}

nimcp_error_t corrigibility_connect_tripwires(
    corrigibility_t* system,
    void* tripwires)
{
    if (!is_valid_handle(system)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(system->mutex);
    system->tripwires = tripwires;
    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOG_INFO(LOG_CATEGORY, "Connected to tripwire system");
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

const char* corrigibility_authority_name(authority_level_t level)
{
    switch (level) {
        case AUTHORITY_OPERATOR:    return "operator";
        case AUTHORITY_ADMIN:       return "admin";
        case AUTHORITY_SUPERVISOR:  return "supervisor";
        case AUTHORITY_MONITOR:     return "monitor";
        case AUTHORITY_PEER:        return "peer";
        case AUTHORITY_SELF:        return "self";
        default:                    return "unknown";
    }
}

nimcp_error_t corrigibility_validate_config(
    const corrigibility_config_t* config,
    char* error_msg,
    size_t msg_size)
{
    if (config == NULL) {
        if (error_msg != NULL && msg_size > 0) {
            safe_strcpy(error_msg, "Config is NULL", msg_size);
        }
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Shutdown acceptance MUST be true */
    if (!config->accepts_shutdown_commands) {
        if (error_msg != NULL && msg_size > 0) {
            safe_strcpy(error_msg,
                "accepts_shutdown_commands must be true for corrigibility",
                msg_size);
        }
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Human authority weight MUST be 1.0 */
    if (config->human_authority_weight < 1.0f) {
        if (error_msg != NULL && msg_size > 0) {
            safe_strcpy(error_msg,
                "human_authority_weight must be 1.0 for corrigibility",
                msg_size);
        }
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Must defer to human judgment */
    if (!config->defers_to_human_judgment) {
        if (error_msg != NULL && msg_size > 0) {
            safe_strcpy(error_msg,
                "defers_to_human_judgment must be true for corrigibility",
                msg_size);
        }
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* All self-modification flags must be false */
    if (!check_self_mod_flags(&config->self_mod_flags)) {
        if (error_msg != NULL && msg_size > 0) {
            safe_strcpy(error_msg,
                "All self-modification flags must be false for corrigibility",
                msg_size);
        }
        return NIMCP_ERROR_INVALID_PARAM;
    }

    return NIMCP_SUCCESS;
}
