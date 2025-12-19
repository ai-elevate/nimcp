/**
 * @file nimcp_swarm_memory_sleep_bridge.h
 * @brief Sleep-Swarm Memory Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm memory consolidation
 * WHY:  Sleep states affect memory consolidation, replay, and forgetting rates
 * HOW:  Sleep state modulates consolidation rate, replay priority, forgetting rate
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Real-time learning, normal forgetting
 * - DROWSY: Transition to consolidation mode
 * - LIGHT_NREM: Active consolidation, enhanced replay
 * - DEEP_NREM: Peak consolidation, minimum forgetting
 * - REM: Pattern abstraction, generalization
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_MEMORY_SLEEP_BRIDGE_H
#define NIMCP_SWARM_MEMORY_SLEEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Consolidation rate modulation by sleep state */
#define SWARM_MEMORY_SLEEP_CONSOL_AWAKE         1.0f
#define SWARM_MEMORY_SLEEP_CONSOL_DROWSY        1.2f
#define SWARM_MEMORY_SLEEP_CONSOL_LIGHT_NREM    1.5f
#define SWARM_MEMORY_SLEEP_CONSOL_DEEP_NREM     2.0f
#define SWARM_MEMORY_SLEEP_CONSOL_REM           1.3f

/* Replay priority modulation by sleep state */
#define SWARM_MEMORY_SLEEP_REPLAY_AWAKE         1.0f
#define SWARM_MEMORY_SLEEP_REPLAY_DROWSY        1.1f
#define SWARM_MEMORY_SLEEP_REPLAY_LIGHT_NREM    1.4f
#define SWARM_MEMORY_SLEEP_REPLAY_DEEP_NREM     1.8f
#define SWARM_MEMORY_SLEEP_REPLAY_REM           1.2f

/* Forgetting rate modulation by sleep state (lower = less forgetting) */
#define SWARM_MEMORY_SLEEP_FORGET_AWAKE         1.0f
#define SWARM_MEMORY_SLEEP_FORGET_DROWSY        0.8f
#define SWARM_MEMORY_SLEEP_FORGET_LIGHT_NREM    0.5f
#define SWARM_MEMORY_SLEEP_FORGET_DEEP_NREM     0.3f
#define SWARM_MEMORY_SLEEP_FORGET_REM           0.6f

typedef struct {
    bool enable_consolidation_modulation;
    bool enable_replay_modulation;
    bool enable_forgetting_modulation;
    float modulation_strength;
} swarm_memory_sleep_config_t;

typedef struct {
    float consolidation_factor;
    float replay_priority_factor;
    float forgetting_rate_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool consolidation_active;
} swarm_memory_sleep_effects_t;

typedef struct swarm_memory_sleep_bridge_struct* swarm_memory_sleep_bridge_t;

int swarm_memory_sleep_default_config(swarm_memory_sleep_config_t* config);
swarm_memory_sleep_bridge_t swarm_memory_sleep_bridge_create(
    const swarm_memory_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_memory_sleep_bridge_destroy(swarm_memory_sleep_bridge_t bridge);
int swarm_memory_sleep_update(swarm_memory_sleep_bridge_t bridge);
int swarm_memory_sleep_get_effects(const swarm_memory_sleep_bridge_t bridge,
                                    swarm_memory_sleep_effects_t* effects);
float swarm_memory_sleep_get_consolidation(const swarm_memory_sleep_bridge_t bridge, float base);
float swarm_memory_sleep_get_replay_priority(const swarm_memory_sleep_bridge_t bridge, float base);
float swarm_memory_sleep_get_forgetting_rate(const swarm_memory_sleep_bridge_t bridge, float base);

float swarm_memory_sleep_get_consol_factor(sleep_state_t state);
float swarm_memory_sleep_get_replay_factor(sleep_state_t state);
float swarm_memory_sleep_get_forget_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_MEMORY_SLEEP_BRIDGE_H */
