/**
 * @file nimcp_energy_consistency_thermo_bridge.c
 * @brief Energy-Thermodynamics Bridge Implementation
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_energy_consistency_thermo_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for energy_consistency_thermo_bridge module */
static nimcp_health_agent_t* g_energy_consistency_thermo_bridge_health_agent = NULL;

/**
 * @brief Set health agent for energy_consistency_thermo_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void energy_consistency_thermo_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_energy_consistency_thermo_bridge_health_agent = agent;
}

/** @brief Send heartbeat from energy_consistency_thermo_bridge module */
static inline void energy_consistency_thermo_bridge_heartbeat(const char* operation, float progress) {
    if (g_energy_consistency_thermo_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_energy_consistency_thermo_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from energy_consistency_thermo_bridge module (instance-level) */
static inline void energy_consistency_thermo_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_energy_consistency_thermo_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_energy_consistency_thermo_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_energy_consistency_thermo_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "ENERGY_CONSISTENCY_THERMO_BRIDGE"


#define BOLTZMANN_CONSTANT 1.380649e-23f

NIMCP_API energy_thermo_bridge_t* energy_thermo_bridge_create(void) {
    energy_thermo_bridge_t* bridge = nimcp_calloc(1, sizeof(energy_thermo_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }

    bridge_base_init(&bridge->base, BIO_MODULE_ENERGY_THERMO_BRIDGE,
                     "energy_thermo_bridge");

    bridge->config.enable_modulation = true;
    bridge->config.atp_per_operation = 0.001f;
    bridge->config.landauer_constant = BOLTZMANN_CONSTANT;
    bridge->config.temperature_kelvin = 310.0f; /* Body temperature */

    NIMCP_LOGGING_INFO("Created %s bridge", "energy_consistency_thermo");
    return bridge;
}

NIMCP_API void energy_thermo_bridge_destroy(energy_thermo_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "energy_consistency_thermo");
    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);
}

NIMCP_API nimcp_error_t energy_thermo_bridge_track_reasoning_cost(
    energy_thermo_bridge_t* bridge, uint32_t operations) {

    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "energy_thermo_bridge_track_reasoning_cost: bridge is NULL");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    bridge->total_atp_consumed += operations * bridge->config.atp_per_operation;
    bridge->operations_tracked += operations;

    return NIMCP_SUCCESS;
}

NIMCP_API double energy_thermo_bridge_landauer_proof_cost(
    energy_thermo_bridge_t* bridge, uint32_t proof_bits) {

    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "energy_thermo_bridge_landauer_proof_cost: bridge is NULL");
        return 0.0;
    }

    /* Landauer limit: E >= k*T*ln(2) per bit erased */
    double min_energy_per_bit = bridge->config.landauer_constant *
                                 bridge->config.temperature_kelvin * log(2.0);

    double total_cost = min_energy_per_bit * proof_bits;
    bridge->total_landauer_cost += (float)total_cost;

    return total_cost;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void energy_consistency_thermo_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_energy_consistency_thermo_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int energy_consistency_thermo_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "energy_consistency_thermo_bridge_training_begin: NULL argument");
        return -1;
    }
    energy_consistency_thermo_bridge_heartbeat_instance(NULL, "energy_consistency_thermo_bridge_training_begin", 0.0f);
    return 0;
}

int energy_consistency_thermo_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "energy_consistency_thermo_bridge_training_end: NULL argument");
        return -1;
    }
    energy_consistency_thermo_bridge_heartbeat_instance(NULL, "energy_consistency_thermo_bridge_training_end", 1.0f);
    return 0;
}

int energy_consistency_thermo_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "energy_consistency_thermo_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    energy_consistency_thermo_bridge_heartbeat_instance(NULL, "energy_consistency_thermo_bridge_training_step", progress);
    return 0;
}
