//=============================================================================
// nimcp_creative_orchestrator.c - Creative System Orchestrator Implementation
//=============================================================================
/**
 * @file nimcp_creative_orchestrator.c
 * @brief Master orchestrator for creative/artistic cognitive system
 *
 * WHAT: Coordinates all creative subsystems for artistic cognition
 * WHY:  Unified interface for appreciation and generation capabilities
 * HOW:  Manages lifecycle and routing between creative components
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-01-30
 */

#include "cognitive/creative/nimcp_creative_orchestrator.h"
#include "cognitive/creative/nimcp_creative.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <time.h>

#define LOG_MODULE "CREATIVE_ORCH"

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for creative orchestrator module */
static nimcp_health_agent_t* g_creative_orchestrator_health_agent = NULL;

/**
 * @brief Set health agent for creative orchestrator heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void creative_orchestrator_set_health_agent(nimcp_health_agent_t* agent) {
    g_creative_orchestrator_health_agent = agent;
}

/** @brief Send heartbeat from creative orchestrator module */
static inline void creative_orchestrator_heartbeat(const char* operation, float progress) {
    if (g_creative_orchestrator_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_creative_orchestrator_health_agent, operation, progress);
    }
}

/** @brief Dual-level heartbeat: instance agent or global */
static inline void creative_orchestrator_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (instance_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    } else if (g_creative_orchestrator_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_creative_orchestrator_health_agent, operation, progress);
    }
}

//=============================================================================
// Lifecycle API
//=============================================================================

creative_orchestrator_t* creative_orchestrator_create(
    const creative_config_t* config) {

    creative_orchestrator_t* orch = nimcp_calloc(1, sizeof(creative_orchestrator_t));
    if (!orch) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate orchestrator");
        return NULL;
    }

    /* Apply config or defaults */
    if (config) {
        memcpy(&orch->config, config, sizeof(creative_config_t));
    } else {
        creative_config_init_defaults(&orch->config);
    }

    /* Initialize statistics */
    memset(&orch->stats, 0, sizeof(creative_orchestrator_stats_t));

    /* Initialize state */
    orch->state = CREATIVE_STATE_INITIALIZING;

    LOG_INFO(LOG_MODULE, "Creative orchestrator created");

    creative_orchestrator_heartbeat("orchestrator_create", 1.0f);

    orch->state = CREATIVE_STATE_READY;
    return orch;
}

void creative_orchestrator_destroy(creative_orchestrator_t* orch) {
    if (!orch) return;

    orch->state = CREATIVE_STATE_SHUTDOWN;

    /* Note: Subsystem destruction would be implemented when subsystems exist */
    /* For now, just free the orchestrator struct */

    nimcp_free(orch);
    LOG_INFO(LOG_MODULE, "Creative orchestrator destroyed");
}

int creative_orchestrator_init_subsystems(creative_orchestrator_t* orch) {
    if (!orch) return -1;

    creative_orchestrator_heartbeat("init_subsystems", 0.0f);

    /* Subsystem initialization would happen here */
    /* Currently a stub for future implementation */

    creative_orchestrator_heartbeat("init_subsystems", 1.0f);

    return 0;
}

void creative_orchestrator_shutdown(creative_orchestrator_t* orch) {
    if (!orch) return;

    orch->state = CREATIVE_STATE_SHUTDOWN;
}

//=============================================================================
// Update API
//=============================================================================

int creative_orchestrator_update(creative_orchestrator_t* orch, uint64_t dt_us) {
    if (!orch) return -1;

    creative_orchestrator_heartbeat("orchestrator_update", 0.0f);

    /* Update statistics */
    orch->stats.update_cycles++;
    orch->stats.last_update_us = dt_us;

    creative_orchestrator_heartbeat("orchestrator_update", 1.0f);

    return 0;
}

creative_orchestrator_state_t creative_orchestrator_get_state(
    const creative_orchestrator_t* orch) {
    if (!orch) return CREATIVE_STATE_UNINITIALIZED;
    return orch->state;
}

int creative_orchestrator_get_stats(const creative_orchestrator_t* orch,
                                     creative_orchestrator_stats_t* out) {
    if (!orch || !out) return -1;
    *out = orch->stats;
    return 0;
}

void creative_orchestrator_reset_stats(creative_orchestrator_t* orch) {
    if (!orch) return;
    memset(&orch->stats, 0, sizeof(creative_orchestrator_stats_t));
}

//=============================================================================
// Brain Integration API
//=============================================================================

void creative_orchestrator_set_brain(creative_orchestrator_t* orch, void* brain) {
    if (!orch) return;
    orch->brain = brain;
}

void creative_orchestrator_set_emotion_system(creative_orchestrator_t* orch,
                                               void* emotion) {
    if (!orch) return;
    orch->emotion_system = emotion;
}

void creative_orchestrator_set_hippocampus(creative_orchestrator_t* orch,
                                            void* hippo) {
    if (!orch) return;
    orch->hippocampus = hippo;
}

void creative_orchestrator_set_ethics_engine(creative_orchestrator_t* orch,
                                              void* ethics) {
    if (!orch) return;
    orch->ethics_engine = ethics;
}

void creative_orchestrator_set_immune_system(creative_orchestrator_t* orch,
                                              void* immune) {
    if (!orch) return;
    orch->immune_system = immune;
}

void creative_orchestrator_set_temporal_lobe(creative_orchestrator_t* orch,
                                              void* temporal) {
    if (!orch) return;
    orch->temporal_lobe = temporal;
}

void creative_orchestrator_set_vae_system(creative_orchestrator_t* orch,
                                           void* vae) {
    if (!orch) return;
    orch->vae_system = vae;
}

//=============================================================================
// Component Accessors
//=============================================================================

aesthetic_evaluator_t* creative_orchestrator_get_aesthetic_eval(
    creative_orchestrator_t* orch) {
    return orch ? orch->aesthetic_eval : NULL;
}

style_representer_t* creative_orchestrator_get_style_repr(
    creative_orchestrator_t* orch) {
    return orch ? orch->style_repr : NULL;
}

text_generator_t* creative_orchestrator_get_text_gen(
    creative_orchestrator_t* orch) {
    return orch ? orch->text_gen : NULL;
}

music_generator_t* creative_orchestrator_get_music_gen(
    creative_orchestrator_t* orch) {
    return orch ? orch->music_gen : NULL;
}

visual_generator_t* creative_orchestrator_get_visual_gen(
    creative_orchestrator_t* orch) {
    return orch ? orch->visual_gen : NULL;
}

diffusion_bridge_t* creative_orchestrator_get_diffusion_bridge(
    creative_orchestrator_t* orch) {
    return orch ? orch->diffusion_bridge : NULL;
}
