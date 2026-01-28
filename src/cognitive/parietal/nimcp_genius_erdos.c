/**
 * @file nimcp_genius_erdos.c
 * @brief Erdős Mode Implementation (Combinatorics & Graph Theory)
 *
 * Implements Erdős-style mathematical reasoning focusing on:
 * - Combinatorics and counting
 * - Graph theory algorithms
 * - Probabilistic method proofs
 *
 * @author NIMCP Team
 * @version 2.6.3
 */

#include "cognitive/parietal/nimcp_mathematical_genius.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <math.h>
#include <string.h>
#include "utils/exception/nimcp_exception_macros.h"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for genius_erdos module */
static nimcp_health_agent_t* g_genius_erdos_health_agent = NULL;

/**
 * @brief Set health agent for genius_erdos heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void genius_erdos_set_health_agent(nimcp_health_agent_t* agent) {
    g_genius_erdos_health_agent = agent;
}

/** @brief Send heartbeat from genius_erdos module */
static inline void genius_erdos_heartbeat(const char* operation, float progress) {
    if (g_genius_erdos_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_genius_erdos_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from genius_erdos module (instance-level) */
static inline void genius_erdos_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_genius_erdos_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_genius_erdos_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_genius_erdos_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


/* ============================================================================
 * Erdős Mode Analysis Implementation
 * ============================================================================ */

nimcp_error_t genius_erdos_analyze_impl(
    mathematical_genius_t* genius,
    const math_problem_t* problem,
    genius_result_t* result) {

    /* Phase 8: Heartbeat at operation start */
    genius_erdos_heartbeat("genius_erdos_analyze_impl", 0.0f);


    (void)genius;  /* Suppress unused warning for now */

    if (!problem || !result) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    uint64_t start = nimcp_time_monotonic_us();

    /* Initialize result */
    memset(result, 0, sizeof(genius_result_t));
    result->mode_used = GENIUS_MODE_ERDOS;

    /* Basic Erdős analysis - placeholder implementation */
    result->elegance_score = 0.9f;
    result->novelty_score = 0.85f;
    result->generalization_score = 0.7f;

    (void)start;  /* Timing tracked elsewhere */

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Ramsey Theory Functions
 * ============================================================================ */

/**
 * @brief Compute lower bound for Ramsey number R(r,s)
 *
 * Uses Erdős probabilistic lower bound: R(r,s) > floor(2^((r+s-2)/2))
 * This is the classic result from Erdős's 1947 probabilistic method proof.
 *
 * @param genius Genius instance (unused, for API consistency)
 * @param r First parameter of R(r,s)
 * @param s Second parameter of R(r,s)
 * @return Lower bound for R(r,s)
 */
uint32_t genius_erdos_ramsey_lower_bound(
    mathematical_genius_t* genius,
    uint32_t r,
    uint32_t s) {

    /* Phase 8: Heartbeat at operation start */
    genius_erdos_heartbeat("genius_erdos_ramsey_lower_bound", 0.0f);


    (void)genius;  /* Unused - function is stateless */

    if (r < 2 || s < 2) {
        return 1;  /* Base cases */
    }

    /* Erdős probabilistic lower bound: R(r,s) > 2^((r+s-2)/2) */
    double exponent = (double)(r + s - 2) / 2.0;
    double lower_bound = pow(2.0, exponent);

    /* Return floor of the bound */
    return (uint32_t)floor(lower_bound);
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void genius_erdos_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_genius_erdos_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int genius_erdos_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "genius_erdos_training_begin: NULL argument");
        return -1;
    }
    genius_erdos_heartbeat_instance(NULL, "genius_erdos_training_begin", 0.0f);
    return 0;
}

int genius_erdos_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "genius_erdos_training_end: NULL argument");
        return -1;
    }
    genius_erdos_heartbeat_instance(NULL, "genius_erdos_training_end", 1.0f);
    return 0;
}

int genius_erdos_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "genius_erdos_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    genius_erdos_heartbeat_instance(NULL, "genius_erdos_training_step", progress);
    return 0;
}
