/**
 * @file nimcp_forward_chaining.c
 * @brief MODULE 3: Forward Chaining Engine implementation
 *
 * SINGLE RESPONSIBILITY: Derive new facts from existing knowledge (data-driven inference)
 *
 * @author NIMCP Development Team - SRP Refactoring
 * @date 2025-11-20
 * @version 3.0.0
 */

#include "cognitive/reasoning/nimcp_forward_chaining.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/reasoning/nimcp_symbolic_logic_attachment.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "core/events/nimcp_event_bus.h"
#include "cognitive/nimcp_working_memory.h"
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
// Event Types
//=============================================================================

#define EVENT_FORWARD_CHAIN_STEP 0x0945
#define EVENT_NOVEL_FACT_DERIVED 0x0946

//=============================================================================
// Configuration
//=============================================================================

#define MAX_ITERATIONS_CAP 1000
#define DEFAULT_SALIENCE 0.7f
#define MAX_WM_FACTS 7  // Working memory capacity limit

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

const char* forward_chain_get_last_error(void)
{
    return last_error;
}

//=============================================================================
// Forward Chaining Implementation
//=============================================================================

bool brain_forward_chain(
    brain_t brain,
    uint32_t max_iterations,
    forward_chain_result_t* result)
{
    // Validate inputs
    if (!nimcp_validate_pointer(brain, "brain")) {
        set_error("Brain is NULL");
        NIMCP_LOGGING_ERROR("brain_forward_chain: brain is NULL");
        return false;
    }

    if (!brain_has_symbolic_logic(brain)) {
        set_error("Symbolic logic engine not attached");
        NIMCP_LOGGING_ERROR("brain_forward_chain: no logic engine attached");
        return false;
    }

    if (!nimcp_validate_pointer(result, "result")) {
        set_error("Result pointer is NULL");
        NIMCP_LOGGING_ERROR("brain_forward_chain: result is NULL");
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(forward_chain_result_t));

    // Cap max iterations
    if (max_iterations == 0 || max_iterations > MAX_ITERATIONS_CAP) {
        max_iterations = MAX_ITERATIONS_CAP;
    }

    // Get logic engine
    symbolic_logic_t* engine = brain_get_symbolic_logic(brain);

    // Create executive task if enabled
    uint32_t task_id = 0;
    // if (brain->executive && brain->config.enable_executive) { // Executive not available
    //     task_descriptor_t task = {
    //         .type = TASK_TYPE_REASONING,
    //         .priority = PRIORITY_NORMAL,
    //         .status = TASK_STATUS_PENDING,
    //         .steps_total = max_iterations
    //     };
    //     strncpy(task.name, "Forward Chaining", sizeof(task.name) - 1);
    //     task_id = executive_add_task(brain->executive, &task);
    // }

    // Perform forward chaining
    uint64_t start_time = nimcp_time_monotonic_ms();
    logic_clause_t** new_facts = NULL;
    int num_new_facts = 0;

    bool success = symbolic_logic_forward_chain(
        engine,
        max_iterations,
        &new_facts,
        &num_new_facts
    );

    uint64_t end_time = nimcp_time_monotonic_ms();

    if (!success) {
        set_error("Forward chaining execution failed");
        NIMCP_LOGGING_ERROR("brain_forward_chain: symbolic_logic_forward_chain failed");
        // if (task_id > 0 && brain->executive) {
        //     executive_mark_task_failed(brain->executive, task_id); // Executive task tracking not available
        // }
        return false;
    }

    // Store new facts in working memory if enabled
    if (brain->working_memory && brain->config.enable_working_memory && num_new_facts > 0) {
        int wm_count = (num_new_facts < MAX_WM_FACTS) ? num_new_facts : MAX_WM_FACTS;
        for (int i = 0; i < wm_count; i++) {
            float fact_encoding[4] = {DEFAULT_SALIENCE, 1.0F, (float)i, 0.0F};
            working_memory_add(brain->working_memory, fact_encoding, 4, DEFAULT_SALIENCE);
        }
        NIMCP_LOGGING_DEBUG("Stored %d new inferences in working memory", wm_count);
    }

    // Publish events for each derived fact
    // if (brain->event_bus) { // Event bus always exists now
    //     for (int i = 0; i < num_new_facts; i++) {
    //         event_data_t event = {
    //             .event_type = EVENT_NOVEL_FACT_DERIVED,
    //             .timestamp = 0,
    //             .priority = EVENT_PRIORITY_NORMAL,
    //             .data_size = sizeof(void*),
    //             .data = (void*)new_facts[i]
    //         };
    //         event_bus_publish(brain->event_bus, &event); // Event API changed
    //     }
    // }

    // Populate result
    result->new_facts = new_facts;
    result->num_new_facts = num_new_facts;
    result->iterations_performed = max_iterations; // Actual iterations may be less
    result->confidence = (num_new_facts > 0) ? 0.8F : 0.0F;
    result->inference_time_ms = end_time - start_time;
    result->converged = (num_new_facts == 0); // Converged if no new facts

    // Complete executive task if enabled
    // if (task_id > 0 && brain->executive) {
    //     executive_mark_task_XXX(XXX); // Executive task tracking not available
    // }

    NIMCP_LOGGING_INFO("Forward chaining completed: %d new facts derived in %llu ms",
                       num_new_facts, (unsigned long long)(end_time - start_time));

    return true;
}

bool brain_forward_chain_step(
    brain_t brain,
    logic_clause_t*** new_facts,
    uint32_t* num_new_facts)
{
    // Validate inputs
    if (!nimcp_validate_pointer(brain, "brain")) {
        set_error("Brain is NULL");
        NIMCP_LOGGING_ERROR("brain_forward_chain_step: brain is NULL");
        return false;
    }

    if (!brain_has_symbolic_logic(brain)) {
        set_error("Symbolic logic engine not attached");
        NIMCP_LOGGING_ERROR("brain_forward_chain_step: no logic engine attached");
        return false;
    }

    if (!nimcp_validate_pointer(new_facts, "new_facts")) {
        set_error("New facts pointer is NULL");
        NIMCP_LOGGING_ERROR("brain_forward_chain_step: new_facts is NULL");
        return false;
    }

    if (!nimcp_validate_pointer(num_new_facts, "num_new_facts")) {
        set_error("Num new facts pointer is NULL");
        NIMCP_LOGGING_ERROR("brain_forward_chain_step: num_new_facts is NULL");
        return false;
    }

    // Get logic engine
    symbolic_logic_t* engine = brain_get_symbolic_logic(brain);

    // Perform single forward chaining step
    int num_derived = 0;
    bool success = symbolic_logic_forward_chain(
        engine,
        1,  // Single iteration
        new_facts,
        &num_derived
    );

    if (!success) {
        set_error("Forward chain step execution failed");
        NIMCP_LOGGING_ERROR("brain_forward_chain_step: execution failed");
        return false;
    }

    *num_new_facts = num_derived;

    // Publish step event
    // if (brain->event_bus) { // Event bus always exists now
    //     event_data_t event = {
    //         .event_type = EVENT_FORWARD_CHAIN_STEP,
    //         .timestamp = 0,
    //         .priority = EVENT_PRIORITY_LOW,
    //         .data_size = sizeof(uint32_t),
    //         .data = (void*)(uintptr_t)num_derived
    //     };
    //     event_bus_publish(brain->event_bus, &event); // Event API changed
    // }

    NIMCP_LOGGING_DEBUG("Forward chain step: %d facts derived", num_derived);
    return true;
}

void forward_chain_free_result(forward_chain_result_t* result)
{
    if (!result) return;

    if (result->new_facts) {
        // Free each fact
        for (uint32_t i = 0; i < result->num_new_facts; i++) {
            if (result->new_facts[i]) {
                nimcp_free(result->new_facts[i]);
            }
        }
        // Free the array
        nimcp_free(result->new_facts);
        result->new_facts = NULL;
    }

    memset(result, 0, sizeof(forward_chain_result_t));
}

bool brain_get_forward_chain_stats(
    brain_t brain,
    uint32_t* iterations,
    uint32_t* facts_derived,
    uint64_t* time_ms)
{
    if (!brain || !brain_has_symbolic_logic(brain)) {
        set_error("Brain is NULL or no logic engine attached");
        return false;
    }

    // Get statistics from logic engine
    logic_stats_t stats;
    symbolic_logic_t* engine = brain_get_symbolic_logic(brain);

    if (!symbolic_logic_get_stats(engine, &stats)) {
        set_error("Failed to get logic engine statistics");
        return false;
    }

    // Populate outputs (if provided)
    if (iterations) {
        *iterations = 0; // Not tracked separately, would need extension
    }

    if (facts_derived) {
        *facts_derived = stats.facts_stored;
    }

    if (time_ms) {
        *time_ms = (uint64_t)(stats.avg_inference_time * stats.inferences_performed);
    }

    return true;
}
