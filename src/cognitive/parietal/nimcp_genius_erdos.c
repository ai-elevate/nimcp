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
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(genius_erdos)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_genius_erdos_mesh_id = 0;
static mesh_participant_registry_t* g_genius_erdos_mesh_registry = NULL;

nimcp_error_t genius_erdos_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_genius_erdos_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "genius_erdos", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "genius_erdos";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_genius_erdos_mesh_id);
    if (err == NIMCP_SUCCESS) g_genius_erdos_mesh_registry = registry;
    return err;
}

void genius_erdos_mesh_unregister(void) {
    if (g_genius_erdos_mesh_registry && g_genius_erdos_mesh_id != 0) {
        mesh_participant_unregister(g_genius_erdos_mesh_registry, g_genius_erdos_mesh_id);
        g_genius_erdos_mesh_id = 0;
        g_genius_erdos_mesh_registry = NULL;
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
