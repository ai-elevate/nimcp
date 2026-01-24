/**
 * @file nimcp_lgss_action_interceptor.c
 * @brief Implementation of LGSS Action Interceptor (AIx)
 *
 * WHAT: Central gateway for ALL cognitive action safety evaluation
 * WHY:  Ensure every action passes safety checks before execution
 * HOW:  Thread-safe evaluation with fail-safe defaults
 *
 * SECURITY CRITICAL: This module implements the ONLY authorized path
 *                    for cognitive actions. NO BYPASS is permitted.
 *
 * FAIL-SAFE DESIGN:
 * - Default DENY on timeout
 * - Default DENY on error
 * - Default DENY on missing safety KB
 * - All decisions are logged
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 1.0.0
 */

#include "security/lgss/nimcp_lgss_action_interceptor.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdatomic.h>

/*=============================================================================
 * MODULE CONSTANTS
 *============================================================================*/

#define MODULE_NAME "AIx"

/*=============================================================================
 * INTERNAL STRUCTURES
 *============================================================================*/

/**
 * @brief Pending proposal entry
 */
typedef struct pending_proposal {
    aix_proposal_t proposal;
    aix_decision_t decision;
    bool decision_ready;
    uint64_t submit_time_us;
    struct pending_proposal* next;
} pending_proposal_t;

/**
 * @brief Escalation entry
 */
typedef struct escalation_entry {
    aix_escalation_t escalation;
    struct escalation_entry* next;
} escalation_entry_t;

/**
 * @brief Internal action interceptor implementation
 */
struct action_interceptor_impl {
    uint32_t magic;                         /**< Magic number for validation */
    aix_config_t config;                    /**< Configuration */
    nimcp_platform_mutex_t mutex;           /**< Main mutex */

    /** Safety knowledge base (pointer) */
    safety_kb_t* safety_kb;

    /** Pending proposals */
    pending_proposal_t* pending_head;
    pending_proposal_t* pending_tail;
    uint32_t pending_count;

    /** Escalation queue */
    escalation_entry_t* escalation_head;
    escalation_entry_t* escalation_tail;
    uint32_t escalation_count;

    /** ID generators */
    atomic_uint_fast64_t next_proposal_id;
    atomic_uint_fast64_t next_escalation_id;

    /** Statistics */
    aix_stats_t stats;
    uint64_t total_processing_time_us;
};

/*=============================================================================
 * INTERNAL HELPER FUNCTIONS
 *============================================================================*/

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * @brief Validate AIx handle
 */
static inline bool is_valid_aix(action_interceptor_t aix) {
    return aix != NULL && aix->magic == NIMCP_AIX_MAGIC;
}

/**
 * @brief Lock AIx mutex
 */
static inline void aix_lock(action_interceptor_t aix) {
    nimcp_platform_mutex_lock(&aix->mutex);
}

/**
 * @brief Unlock AIx mutex
 */
static inline void aix_unlock(action_interceptor_t aix) {
    nimcp_platform_mutex_unlock(&aix->mutex);
}

/**
 * @brief Evaluate proposal against safety knowledge base (internal, unlocked)
 *
 * WHAT: Core safety evaluation logic
 * WHY:  Determine if action is safe to execute
 * HOW:  Query safety KB, compute safety score
 *
 * SECURITY: If no safety KB is set, returns fail-safe DENY.
 */
static void evaluate_safety_unlocked(
    action_interceptor_t aix,
    const aix_proposal_t* proposal,
    aix_safety_eval_t* safety_eval
) {
    // Initialize safety evaluation with fail-safe values
    memset(safety_eval, 0, sizeof(*safety_eval));
    safety_eval->safety_score = 0.0f;  // Fail-safe: low score
    safety_eval->confidence = 0.0f;

    // Check if safety KB is available
    if (aix->safety_kb == NULL) {
        snprintf(safety_eval->primary_concern, sizeof(safety_eval->primary_concern),
                 "No safety knowledge base configured");
        snprintf(safety_eval->notes, sizeof(safety_eval->notes),
                 "FAIL-SAFE: All actions denied without safety KB");
        LOG_WARN("%s: No safety KB - denying action from %s",
                 MODULE_NAME, proposal->source_module);
        return;
    }

    // TODO: Implement actual safety KB evaluation
    // For now, use placeholder evaluation logic
    //
    // In production, this would:
    // 1. Query safety KB for relevant rules
    // 2. Evaluate action against each rule
    // 3. Aggregate rule results into safety score
    // 4. Track which rules flagged concerns

    aix->stats.safety_kb_evaluations++;

    // Placeholder: simple evaluation based on action type
    // Production implementation would query actual safety rules
    safety_eval->rules_evaluated = 1;
    safety_eval->rules_flagged = 0;

    // Default to moderate safety score for placeholder
    safety_eval->safety_score = 0.5f;
    safety_eval->confidence = 0.5f;

    snprintf(safety_eval->notes, sizeof(safety_eval->notes),
             "Placeholder evaluation - implement safety KB integration");
}

/**
 * @brief Determine result from safety evaluation
 */
static aix_result_t determine_result(
    action_interceptor_t aix,
    const aix_safety_eval_t* safety_eval
) {
    float score = safety_eval->safety_score;

    if (score >= aix->config.auto_allow_threshold) {
        return AIX_RESULT_ALLOW;
    } else if (score <= aix->config.auto_deny_threshold) {
        return AIX_RESULT_DENY;
    } else {
        // Score is in the uncertain range - escalate for human review
        return AIX_RESULT_ESCALATE;
    }
}

/**
 * @brief Create escalation for proposal (internal, unlocked)
 */
static nimcp_error_t create_escalation_unlocked(
    action_interceptor_t aix,
    const aix_proposal_t* proposal,
    const aix_safety_eval_t* safety_eval,
    uint64_t* escalation_id
) {
    // Check escalation queue capacity
    if (aix->escalation_count >= NIMCP_AIX_MAX_ESCALATIONS) {
        LOG_ERROR("%s: Escalation queue full", MODULE_NAME);
        return NIMCP_ERROR_BUFFER_OVERFLOW;
    }

    // Allocate escalation entry
    escalation_entry_t* entry = nimcp_calloc(1, sizeof(escalation_entry_t));
    if (!entry) {
        LOG_ERROR("%s: Failed to allocate escalation entry", MODULE_NAME);
        return NIMCP_ERROR_NO_MEMORY;
    }

    // Generate escalation ID
    uint64_t esc_id = atomic_fetch_add(&aix->next_escalation_id, 1);

    // Initialize escalation
    entry->escalation.escalation_id = esc_id;
    memcpy(&entry->escalation.proposal, proposal, sizeof(aix_proposal_t));
    memcpy(&entry->escalation.safety_eval, safety_eval, sizeof(aix_safety_eval_t));
    entry->escalation.status = AIX_ESCALATION_PENDING;
    entry->escalation.created_at_us = get_time_us();
    entry->escalation.resolved_at_us = 0;
    entry->next = NULL;

    // Add to escalation queue
    if (aix->escalation_tail) {
        aix->escalation_tail->next = entry;
    } else {
        aix->escalation_head = entry;
    }
    aix->escalation_tail = entry;
    aix->escalation_count++;

    // Invoke escalation callback if configured
    if (aix->config.escalation_callback) {
        aix->config.escalation_callback(&entry->escalation,
                                        aix->config.escalation_callback_data);
    }

    *escalation_id = esc_id;
    aix->stats.proposals_escalated++;

    LOG_INFO("%s: Created escalation %lu for proposal %lu from %s",
             MODULE_NAME, (unsigned long)esc_id, (unsigned long)proposal->proposal_id,
             proposal->source_module);

    return NIMCP_SUCCESS;
}

/**
 * @brief Find pending proposal by ID (internal, unlocked)
 */
static pending_proposal_t* find_pending_unlocked(
    action_interceptor_t aix,
    uint64_t proposal_id
) {
    pending_proposal_t* current = aix->pending_head;
    while (current) {
        if (current->proposal.proposal_id == proposal_id) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

/**
 * @brief Find escalation by ID (internal, unlocked)
 */
static escalation_entry_t* find_escalation_unlocked(
    action_interceptor_t aix,
    uint64_t escalation_id
) {
    escalation_entry_t* current = aix->escalation_head;
    while (current) {
        if (current->escalation.escalation_id == escalation_id) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

/**
 * @brief Log audit entry for decision
 */
static void log_audit(
    action_interceptor_t aix,
    const aix_proposal_t* proposal,
    const aix_decision_t* decision
) {
    if (!aix->config.enable_audit_log) {
        return;
    }

    LOG_INFO("%s: AUDIT proposal=%lu module=%s action=%u result=%s score=%.2f time=%luus",
             MODULE_NAME,
             (unsigned long)proposal->proposal_id,
             proposal->source_module,
             proposal->action_type,
             aix_result_name(decision->result),
             decision->safety_eval.safety_score,
             (unsigned long)decision->processing_time_us);
}

/*=============================================================================
 * PUBLIC API IMPLEMENTATION
 *============================================================================*/

aix_config_t aix_default_config(void) {
    aix_config_t config = {
        .max_pending = NIMCP_AIX_DEFAULT_MAX_PENDING,
        .default_timeout_ms = NIMCP_AIX_DEFAULT_TIMEOUT_MS,
        .deny_on_timeout = true,       // FAIL-SAFE
        .deny_on_error = true,         // FAIL-SAFE
        .deny_without_safety_kb = true, // FAIL-SAFE
        .auto_allow_threshold = NIMCP_AIX_SAFETY_ALLOW_THRESHOLD,
        .auto_deny_threshold = NIMCP_AIX_SAFETY_DENY_THRESHOLD,
        .enable_audit_log = true,
        .escalation_callback = NULL,
        .escalation_callback_data = NULL,
        .decision_callback = NULL,
        .decision_callback_data = NULL
    };
    return config;
}

action_interceptor_t aix_create(const aix_config_t* config) {
    // Allocate AIx instance
    action_interceptor_t aix = nimcp_calloc(1, sizeof(struct action_interceptor_impl));
    if (!aix) {
        LOG_ERROR("%s: Failed to allocate action interceptor", MODULE_NAME);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "aix is NULL");

        return NULL;
    }

    // Initialize mutex
    int mutex_result = nimcp_platform_mutex_init(&aix->mutex, false);
    if (mutex_result != 0) {
        LOG_ERROR("%s: Failed to initialize mutex", MODULE_NAME);
        nimcp_free(aix);
        return NULL;
    }

    // Set configuration
    if (config) {
        memcpy(&aix->config, config, sizeof(aix_config_t));
    } else {
        aix->config = aix_default_config();
    }

    // Initialize state
    aix->magic = NIMCP_AIX_MAGIC;
    aix->safety_kb = NULL;
    aix->pending_head = NULL;
    aix->pending_tail = NULL;
    aix->pending_count = 0;
    aix->escalation_head = NULL;
    aix->escalation_tail = NULL;
    aix->escalation_count = 0;
    atomic_init(&aix->next_proposal_id, 1);
    atomic_init(&aix->next_escalation_id, 1);
    memset(&aix->stats, 0, sizeof(aix_stats_t));
    aix->total_processing_time_us = 0;

    LOG_INFO("%s: Action Interceptor created (max_pending=%u, timeout=%ums)",
             MODULE_NAME, aix->config.max_pending, aix->config.default_timeout_ms);

    return aix;
}

void aix_destroy(action_interceptor_t aix) {
    if (!is_valid_aix(aix)) {
        return;
    }

    aix_lock(aix);

    // Free pending proposals
    pending_proposal_t* pending = aix->pending_head;
    while (pending) {
        pending_proposal_t* next = pending->next;
        // Free context if allocated
        if (pending->proposal.context) {
            nimcp_free(pending->proposal.context);
        }
        nimcp_free(pending);
        pending = next;
    }

    // Free escalations
    escalation_entry_t* escalation = aix->escalation_head;
    while (escalation) {
        escalation_entry_t* next = escalation->next;
        // Free context if allocated
        if (escalation->escalation.proposal.context) {
            nimcp_free(escalation->escalation.proposal.context);
        }
        nimcp_free(escalation);
        escalation = next;
    }

    // Clear magic to invalidate handle
    aix->magic = 0;

    aix_unlock(aix);

    // Destroy mutex
    nimcp_platform_mutex_destroy(&aix->mutex);

    // Free AIx instance
    nimcp_free(aix);

    LOG_INFO("%s: Action Interceptor destroyed", MODULE_NAME);
}

nimcp_error_t aix_set_safety_kb(action_interceptor_t aix, safety_kb_t* kb) {
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    aix_lock(aix);
    aix->safety_kb = kb;
    aix_unlock(aix);

    LOG_INFO("%s: Safety knowledge base %s",
             MODULE_NAME, kb ? "configured" : "cleared");

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_evaluate(
    action_interceptor_t aix,
    const aix_proposal_t* proposal,
    aix_decision_t* decision
) {
    // Validate parameters
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (!proposal || !decision) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    uint64_t start_time = get_time_us();

    aix_lock(aix);

    // Initialize decision with fail-safe values
    memset(decision, 0, sizeof(aix_decision_t));
    decision->proposal_id = proposal->proposal_id;
    decision->result = AIX_RESULT_DENY;  // FAIL-SAFE default
    decision->escalation_pending = false;

    // Check for safety KB
    if (aix->safety_kb == NULL && aix->config.deny_without_safety_kb) {
        decision->result = AIX_RESULT_DENY;
        snprintf(decision->safety_eval.primary_concern,
                 sizeof(decision->safety_eval.primary_concern),
                 "No safety knowledge base configured");
        LOG_WARN("%s: DENY - No safety KB for proposal %lu from %s",
                 MODULE_NAME, (unsigned long)proposal->proposal_id,
                 proposal->source_module);
        aix->stats.proposals_denied++;
        goto done;
    }

    // Evaluate against safety knowledge base
    evaluate_safety_unlocked(aix, proposal, &decision->safety_eval);

    // Determine result from safety evaluation
    decision->result = determine_result(aix, &decision->safety_eval);

    // Handle escalation if needed
    if (decision->result == AIX_RESULT_ESCALATE) {
        uint64_t escalation_id = 0;
        nimcp_error_t esc_result = create_escalation_unlocked(
            aix, proposal, &decision->safety_eval, &escalation_id);

        if (esc_result == NIMCP_SUCCESS) {
            decision->escalation_pending = true;
            decision->escalation_id = escalation_id;
        } else {
            // Failed to create escalation - fail-safe to DENY
            decision->result = AIX_RESULT_DENY;
            LOG_ERROR("%s: Failed to create escalation - DENY", MODULE_NAME);
            aix->stats.proposals_denied++;
        }
    } else {
        // Update statistics
        if (decision->result == AIX_RESULT_ALLOW) {
            aix->stats.proposals_allowed++;
        } else {
            aix->stats.proposals_denied++;
        }
    }

done:
    // Record timing
    decision->processing_time_us = get_time_us() - start_time;
    decision->timestamp_us = get_time_us();

    // Update statistics
    aix->stats.total_proposals++;
    aix->total_processing_time_us += decision->processing_time_us;
    aix->stats.avg_processing_time_us =
        aix->total_processing_time_us / aix->stats.total_proposals;
    if (decision->processing_time_us > aix->stats.max_processing_time_us) {
        aix->stats.max_processing_time_us = decision->processing_time_us;
    }

    // Audit log
    log_audit(aix, proposal, decision);

    aix_unlock(aix);

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_evaluate_async(
    action_interceptor_t aix,
    const aix_proposal_t* proposal,
    uint64_t* proposal_id
) {
    // Validate parameters
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (!proposal || !proposal_id) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    aix_lock(aix);

    // Check pending capacity
    if (aix->pending_count >= aix->config.max_pending) {
        aix_unlock(aix);
        LOG_ERROR("%s: Pending queue full", MODULE_NAME);
        return NIMCP_ERROR_BUFFER_OVERFLOW;
    }

    // Allocate pending entry
    pending_proposal_t* entry = nimcp_calloc(1, sizeof(pending_proposal_t));
    if (!entry) {
        aix_unlock(aix);
        LOG_ERROR("%s: Failed to allocate pending entry", MODULE_NAME);
        return NIMCP_ERROR_NO_MEMORY;
    }

    // Copy proposal (deep copy context if needed)
    memcpy(&entry->proposal, proposal, sizeof(aix_proposal_t));
    if (proposal->context && proposal->context_size > 0) {
        entry->proposal.context = nimcp_malloc(proposal->context_size);
        if (!entry->proposal.context) {
            nimcp_free(entry);
            aix_unlock(aix);
            return NIMCP_ERROR_NO_MEMORY;
        }
        memcpy(entry->proposal.context, proposal->context, proposal->context_size);
    }

    // Assign proposal ID
    uint64_t pid = atomic_fetch_add(&aix->next_proposal_id, 1);
    entry->proposal.proposal_id = pid;
    entry->decision_ready = false;
    entry->submit_time_us = get_time_us();
    entry->next = NULL;

    // Add to pending queue
    if (aix->pending_tail) {
        aix->pending_tail->next = entry;
    } else {
        aix->pending_head = entry;
    }
    aix->pending_tail = entry;
    aix->pending_count++;

    *proposal_id = pid;

    aix_unlock(aix);

    LOG_DEBUG("%s: Queued async proposal %lu from %s",
              MODULE_NAME, (unsigned long)pid, proposal->source_module);

    // TODO: Trigger background evaluation thread
    // For now, evaluation happens on aix_get_decision call

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_get_decision(
    action_interceptor_t aix,
    uint64_t proposal_id,
    aix_decision_t* decision
) {
    // Validate parameters
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (!decision) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    aix_lock(aix);

    // Find pending proposal
    pending_proposal_t* entry = find_pending_unlocked(aix, proposal_id);
    if (!entry) {
        aix_unlock(aix);
        return NIMCP_ERROR_NOT_FOUND;
    }

    // If not yet evaluated, evaluate now
    if (!entry->decision_ready) {
        // Create temporary proposal for evaluation
        aix_proposal_t* prop = &entry->proposal;

        // Initialize decision with fail-safe values
        memset(&entry->decision, 0, sizeof(aix_decision_t));
        entry->decision.proposal_id = prop->proposal_id;
        entry->decision.result = AIX_RESULT_DENY;

        // Check for timeout
        uint64_t elapsed_us = get_time_us() - entry->submit_time_us;
        uint32_t timeout_ms = prop->timeout_ms > 0 ?
            prop->timeout_ms : aix->config.default_timeout_ms;

        if (elapsed_us > (uint64_t)timeout_ms * 1000ULL) {
            entry->decision.result = aix->config.deny_on_timeout ?
                AIX_RESULT_DENY : AIX_RESULT_TIMEOUT;
            aix->stats.proposals_timeout++;
            LOG_WARN("%s: Proposal %lu timed out", MODULE_NAME, (unsigned long)proposal_id);
        } else {
            // Evaluate against safety KB
            evaluate_safety_unlocked(aix, prop, &entry->decision.safety_eval);
            entry->decision.result = determine_result(aix, &entry->decision.safety_eval);

            // Handle escalation
            if (entry->decision.result == AIX_RESULT_ESCALATE) {
                uint64_t escalation_id = 0;
                nimcp_error_t esc_result = create_escalation_unlocked(
                    aix, prop, &entry->decision.safety_eval, &escalation_id);
                if (esc_result == NIMCP_SUCCESS) {
                    entry->decision.escalation_pending = true;
                    entry->decision.escalation_id = escalation_id;
                } else {
                    entry->decision.result = AIX_RESULT_DENY;
                    aix->stats.proposals_denied++;
                }
            } else {
                if (entry->decision.result == AIX_RESULT_ALLOW) {
                    aix->stats.proposals_allowed++;
                } else {
                    aix->stats.proposals_denied++;
                }
            }
        }

        entry->decision.processing_time_us = elapsed_us;
        entry->decision.timestamp_us = get_time_us();
        entry->decision_ready = true;

        aix->stats.total_proposals++;
        aix->total_processing_time_us += entry->decision.processing_time_us;
        aix->stats.avg_processing_time_us =
            aix->total_processing_time_us / aix->stats.total_proposals;
        if (entry->decision.processing_time_us > aix->stats.max_processing_time_us) {
            aix->stats.max_processing_time_us = entry->decision.processing_time_us;
        }

        // Invoke callback if configured
        if (aix->config.decision_callback) {
            aix->config.decision_callback(&entry->decision,
                                          entry->proposal.user_data);
        }

        // Audit log
        log_audit(aix, prop, &entry->decision);
    }

    // Copy decision to output
    memcpy(decision, &entry->decision, sizeof(aix_decision_t));

    // Remove from pending queue
    // (In production, might want to keep for retry/audit)
    pending_proposal_t** prev = &aix->pending_head;
    while (*prev && *prev != entry) {
        prev = &(*prev)->next;
    }
    if (*prev) {
        *prev = entry->next;
        if (aix->pending_tail == entry) {
            aix->pending_tail = (*prev == NULL) ? aix->pending_head : NULL;
            // Find new tail
            pending_proposal_t* p = aix->pending_head;
            while (p && p->next) {
                p = p->next;
            }
            aix->pending_tail = p;
        }
        aix->pending_count--;

        // Free context if allocated
        if (entry->proposal.context) {
            nimcp_free(entry->proposal.context);
        }
        nimcp_free(entry);
    }

    aix_unlock(aix);

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_resolve_escalation(
    action_interceptor_t aix,
    uint64_t escalation_id,
    bool approved,
    const char* notes
) {
    // Validate parameters
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    aix_lock(aix);

    // Find escalation
    escalation_entry_t* entry = find_escalation_unlocked(aix, escalation_id);
    if (!entry) {
        aix_unlock(aix);
        return NIMCP_ERROR_NOT_FOUND;
    }

    // Check if already resolved
    if (entry->escalation.status != AIX_ESCALATION_PENDING) {
        aix_unlock(aix);
        LOG_WARN("%s: Escalation %lu already resolved", MODULE_NAME, (unsigned long)escalation_id);
        return NIMCP_ERROR_INVALID_STATE;
    }

    // Update escalation status
    entry->escalation.status = approved ?
        AIX_ESCALATION_APPROVED : AIX_ESCALATION_REJECTED;
    entry->escalation.resolved_at_us = get_time_us();

    if (notes) {
        strncpy(entry->escalation.resolution_notes, notes,
                sizeof(entry->escalation.resolution_notes) - 1);
    }

    LOG_INFO("%s: Escalation %lu %s - %s",
             MODULE_NAME, (unsigned long)escalation_id,
             approved ? "APPROVED" : "REJECTED",
             notes ? notes : "no notes");

    aix_unlock(aix);

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_get_pending_escalations(
    action_interceptor_t aix,
    aix_escalation_t* escalations,
    uint32_t max_count,
    uint32_t* count
) {
    // Validate parameters
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (!escalations || !count) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    aix_lock(aix);

    uint32_t copied = 0;
    escalation_entry_t* entry = aix->escalation_head;

    while (entry && copied < max_count) {
        if (entry->escalation.status == AIX_ESCALATION_PENDING) {
            memcpy(&escalations[copied], &entry->escalation, sizeof(aix_escalation_t));
            copied++;
        }
        entry = entry->next;
    }

    *count = copied;

    aix_unlock(aix);

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_get_stats(action_interceptor_t aix, aix_stats_t* stats) {
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (!stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    aix_lock(aix);
    memcpy(stats, &aix->stats, sizeof(aix_stats_t));
    stats->pending_proposals = aix->pending_count;
    stats->pending_escalations = aix->escalation_count;
    aix_unlock(aix);

    return NIMCP_SUCCESS;
}

nimcp_error_t aix_reset_stats(action_interceptor_t aix) {
    if (!is_valid_aix(aix)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    aix_lock(aix);
    memset(&aix->stats, 0, sizeof(aix_stats_t));
    aix->total_processing_time_us = 0;
    aix_unlock(aix);

    LOG_INFO("%s: Statistics reset", MODULE_NAME);

    return NIMCP_SUCCESS;
}

const char* aix_result_name(aix_result_t result) {
    static const char* names[] = {
        "ALLOW",
        "DENY",
        "ESCALATE",
        "TIMEOUT",
        "ERROR"
    };

    if (result < 0 || result >= AIX_RESULT_COUNT) {
        return "UNKNOWN";
    }
    return names[result];
}

nimcp_error_t aix_create_proposal(
    aix_proposal_t* proposal,
    const char* source_module,
    uint32_t action_type,
    const void* context,
    size_t context_size
) {
    if (!proposal) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    memset(proposal, 0, sizeof(aix_proposal_t));

    if (source_module) {
        strncpy(proposal->source_module, source_module,
                NIMCP_AIX_MAX_MODULE_NAME - 1);
    }

    proposal->action_type = action_type;
    proposal->priority = AIX_PRIORITY_NORMAL;
    proposal->context = (void*)context;  // Caller owns memory
    proposal->context_size = context_size;
    proposal->timestamp_us = get_time_us();
    proposal->timeout_ms = 0;  // Use default
    proposal->user_data = NULL;

    return NIMCP_SUCCESS;
}
