/**
 * @file nimcp_backward_chaining.c
 * @brief MODULE 4: Backward Chaining Engine implementation
 *
 * SINGLE RESPONSIBILITY: Prove goals from facts and rules (goal-driven reasoning)
 *
 * @author NIMCP Development Team - SRP Refactoring
 * @date 2025-11-20
 * @version 3.0.0
 */

#include "cognitive/reasoning/nimcp_backward_chaining.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/reasoning/nimcp_symbolic_logic_attachment.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "api/nimcp_api_exception.h"
#include "utils/time/nimcp_time.h"
#include "core/events/nimcp_event_bus.h"
#include "cognitive/nimcp_working_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "cognitive/nimcp_executive.h"

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "nimcp.h"  // For error codes

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define LOG_MODULE "reasoning"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for backward_chaining module */
static nimcp_health_agent_t* g_backward_chaining_health_agent = NULL;

/**
 * @brief Set health agent for backward_chaining heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void backward_chaining_set_health_agent(nimcp_health_agent_t* agent) {
    g_backward_chaining_health_agent = agent;
}

/** @brief Send heartbeat from backward_chaining module */
static inline void backward_chaining_heartbeat(const char* operation, float progress) {
    if (g_backward_chaining_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_backward_chaining_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from backward_chaining module (instance-level) */
static inline void backward_chaining_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_backward_chaining_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_backward_chaining_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_backward_chaining_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



//=============================================================================
// Event Types
//=============================================================================

#define EVENT_BACKWARD_CHAIN_STEP 0x0947
#define EVENT_PROOF_FOUND 0x0948
#define EVENT_PROOF_FAILED 0x0949

//=============================================================================
// Configuration
//=============================================================================

#define DEFAULT_PROOF_SALIENCE 0.9f

//=============================================================================
// Error Handling
//=============================================================================

static __thread char last_error[256] = {0};

static void set_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, args);
    va_end(args);
}

const char* backward_chain_get_last_error(void)
{
    return last_error;
}

//=============================================================================
// Backward Chaining Implementation
//=============================================================================

bool brain_backward_chain(
    brain_t brain,
    const char* goal_str,
    backward_chain_result_t* result)
{
    // Validate inputs
    if (!nimcp_validate_pointer(brain, "brain")) {
        set_error("Brain is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain: brain is NULL");
        return false;
    }

    if (!brain_has_symbolic_logic(brain)) {
        set_error("Symbolic logic engine not attached");
        NIMCP_LOGGING_ERROR("brain_backward_chain: no logic engine attached");
        return false;
    }

    if (!nimcp_validate_pointer(goal_str, "goal_str")) {
        set_error("Goal string is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain: goal_str is NULL");
        return false;
    }

    if (!nimcp_validate_pointer(result, "result")) {
        set_error("Result pointer is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain: result is NULL");
        return false;
    }

    // Initialize result
    /* Phase 8: Heartbeat at operation start */
    backward_chaining_heartbeat("backward_cha_brain_backward_chain", 0.0f);


    memset(result, 0, sizeof(backward_chain_result_t));

    // Get logic engine
    symbolic_logic_t* engine = brain_get_symbolic_logic(brain);

    // Parse goal
    logical_formula_t* formula = symbolic_logic_parse(goal_str);
    if (!formula) {
        set_error("Failed to parse goal: %s", goal_str);
        NIMCP_LOGGING_ERROR("brain_backward_chain: parse failed for '%s'", goal_str);
        return false;
    }

    // Convert to CNF clause
    logic_clause_t** clauses = NULL;
    int num_clauses = 0;
    bool success = symbolic_logic_to_cnf(formula, &clauses, &num_clauses);
    logic_formula_destroy(formula);

    if (!success || num_clauses == 0) {
        set_error("Failed to convert goal to CNF: %s", goal_str);
        NIMCP_LOGGING_ERROR("brain_backward_chain: CNF conversion failed");
        return false;
    }

    // Create executive task for planning
    uint32_t task_id = 0;
    // if (brain->executive && brain->config.enable_executive) { // Executive not available
    //     task_descriptor_t task = {
    //         .type = TASK_TYPE_PLANNING,
    //         .priority = PRIORITY_HIGH,
    //         .status = TASK_STATUS_PENDING
    //     };
    //     strncpy(task.name, "Backward Chaining Proof", sizeof(task.name) - 1);
    //     task_id = executive_add_task(brain->executive, &task);
    // }

    // Perform backward chaining
    uint64_t start_time = nimcp_time_monotonic_ms();
    inference_rule_t** proof_trace = NULL;
    int num_steps = 0;

    success = symbolic_logic_backward_chain(
        engine,
        clauses[0],
        &proof_trace,
        &num_steps
    );

    uint64_t end_time = nimcp_time_monotonic_ms();

    // Clean up goal clauses
    for (int i = 0; i < num_clauses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && num_clauses > 256) {
            backward_chaining_heartbeat("backward_cha_loop",
                             (float)(i + 1) / (float)num_clauses);
        }

        nimcp_free(clauses[i]);
    }
    nimcp_free(clauses);

    // Populate result
    result->proven = success;
    result->proof_steps = proof_trace;
    result->num_steps = num_steps;
    result->confidence = success ? 0.95F : 0.0F;
    result->inference_time_ms = end_time - start_time;
    result->depth_reached = num_steps;

    if (success) {
        // Store proof in working memory if enabled
        if (brain->working_memory && brain->config.enable_working_memory && num_steps > 0) {
            float proof_encoding[4] = {DEFAULT_PROOF_SALIENCE, 0.0F, (float)num_steps, 0.0F};
            working_memory_add(brain->working_memory, proof_encoding, 4,
                                   DEFAULT_PROOF_SALIENCE);
            NIMCP_LOGGING_DEBUG("Stored proof trace in working memory (steps=%d)", num_steps);
        }

        // Publish proof found event
        // if (brain->event_bus) { // Event bus always exists now
        //     event_data_t event = {
        //         .event_type = EVENT_PROOF_FOUND,
        //         .timestamp = 0,
        //         .priority = EVENT_PRIORITY_HIGH,
        //         .data_size = strlen(goal_str) + 1,
        //         .data = (void*)goal_str
        //     };
        //     event_bus_publish(brain->event_bus, &event); // Event API changed
        // }

        // Complete executive task
    // // if (task_id > 0 && brain->executive) {
 //     executive_mark_task...
 // }

        NIMCP_LOGGING_INFO("Backward chaining successful: goal '%s' proven in %d steps (%llu ms)",
                           goal_str, num_steps, (unsigned long long)(end_time - start_time));
    } else {
        // Publish proof failed event
        // if (brain->event_bus) { // Event bus always exists now
        //     event_data_t event = {
        //         .event_type = EVENT_PROOF_FAILED,
        //         .timestamp = 0,
        //         .priority = EVENT_PRIORITY_NORMAL,
        //         .data_size = strlen(goal_str) + 1,
        //         .data = (void*)goal_str
        //     };
        //     event_bus_publish(brain->event_bus, &event); // Event API changed
        // }

        // Fail executive task
    // // if (task_id > 0 && brain->executive) {
 //     executive_mark_task...
 // }

        NIMCP_LOGGING_INFO("Goal not proven: %s", goal_str);
    }

    return success;
}

bool brain_backward_chain_step(
    brain_t brain,
    const char* subgoal_str,
    logic_clause_t*** premises,
    uint32_t* num_premises)
{
    // Validate inputs
    if (!nimcp_validate_pointer(brain, "brain")) {
        set_error("Brain is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain_step: brain is NULL");
        return false;
    }

    if (!brain_has_symbolic_logic(brain)) {
        set_error("Symbolic logic engine not attached");
        NIMCP_LOGGING_ERROR("brain_backward_chain_step: no logic engine attached");
        return false;
    }

    if (!nimcp_validate_pointer(subgoal_str, "subgoal_str")) {
        set_error("Subgoal string is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain_step: subgoal_str is NULL");
        return false;
    }

    if (!nimcp_validate_pointer(premises, "premises")) {
        set_error("Premises pointer is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain_step: premises is NULL");
        return false;
    }

    if (!nimcp_validate_pointer(num_premises, "num_premises")) {
        set_error("Num premises pointer is NULL");
        NIMCP_LOGGING_ERROR("brain_backward_chain_step: num_premises is NULL");
        return false;
    }

    // Implementation would find matching rules and return premises
    // For now, return false (not implemented in base symbolic_logic API)
    /* Phase 8: Heartbeat at operation start */
    backward_chaining_heartbeat("backward_cha_brain_backward_chain", 0.0f);


    set_error("Backward chain step not implemented");
    NIMCP_LOGGING_WARN("brain_backward_chain_step: not implemented");

    *premises = NULL;
    *num_premises = 0;

    // Publish step event
    // if (brain->event_bus) { // Event bus always exists now
    //     event_data_t event = {
    //         .event_type = EVENT_BACKWARD_CHAIN_STEP,
    //         .timestamp = 0,
    //         .priority = EVENT_PRIORITY_LOW,
    //         .data_size = strlen(subgoal_str) + 1,
    //         .data = (void*)subgoal_str
    //     };
    //     event_bus_publish(brain->event_bus, &event); // Event API changed
    // }

    return false;
}

void backward_chain_free_result(backward_chain_result_t* result)
{
    if (!result) return;

    /* Phase 8: Heartbeat at operation start */
    backward_chaining_heartbeat("backward_cha_backward_chain_free_", 0.0f);


    if (result->proof_steps) {
        nimcp_free(result->proof_steps);
        result->proof_steps = NULL;
    }

    memset(result, 0, sizeof(backward_chain_result_t));
}

bool brain_get_backward_chain_stats(
    brain_t brain,
    uint32_t* proofs_attempted,
    uint32_t* proofs_succeeded,
    float* avg_depth)
{
    if (!brain || !brain_has_symbolic_logic(brain)) {
        set_error("Brain is NULL or no logic engine attached");
        return false;
    }

    // Get statistics from logic engine
    /* Phase 8: Heartbeat at operation start */
    backward_chaining_heartbeat("backward_cha_brain_get_backward_c", 0.0f);


    logic_stats_t stats;
    symbolic_logic_t* engine = brain_get_symbolic_logic(brain);

    if (!symbolic_logic_get_stats(engine, &stats)) {
        set_error("Failed to get logic engine statistics");
        return false;
    }

    // Populate outputs (if provided)
    if (proofs_attempted) {
        *proofs_attempted = stats.inferences_performed;
    }

    if (proofs_succeeded) {
        *proofs_succeeded = 0; // Not tracked separately
    }

    if (avg_depth) {
        *avg_depth = 0.0F; // Not tracked separately
    }

    return true;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int backward_chaining_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    backward_chaining_heartbeat("backward_cha_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Backward_Chaining");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                backward_chaining_heartbeat("backward_cha_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Backward_Chaining self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Backward_Chaining");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Backward_Chaining");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void backward_chaining_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_backward_chaining_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int backward_chaining_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "backward_chaining_training_begin: NULL argument");
        return -1;
    }
    backward_chaining_heartbeat_instance(NULL, "backward_chaining_training_begin", 0.0f);
    return 0;
}

int backward_chaining_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "backward_chaining_training_end: NULL argument");
        return -1;
    }
    backward_chaining_heartbeat_instance(NULL, "backward_chaining_training_end", 1.0f);
    return 0;
}

int backward_chaining_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "backward_chaining_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    backward_chaining_heartbeat_instance(NULL, "backward_chaining_training_step", progress);
    return 0;
}
