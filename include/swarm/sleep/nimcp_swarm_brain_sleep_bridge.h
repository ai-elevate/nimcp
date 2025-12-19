/**
 * @file nimcp_swarm_brain_sleep_bridge.h
 * @brief Sleep-Swarm Brain Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm brain coordination
 * WHY:  Sleep states fundamentally alter swarm coordination dynamics and cognitive capabilities
 * HOW:  Sleep state modulates coordination strength, message frequency, and collective behaviors
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full swarm coordination (active flocking/schooling)
 * - DROWSY: Reduced coordination (settling behavior)
 * - LIGHT_NREM: Minimal coordination (isolated resting)
 * - DEEP_NREM: No coordination (deep sleep state)
 * - REM: Sporadic coordination (sleep twitching, minimal signaling)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_BRAIN_SLEEP_BRIDGE_H
#define NIMCP_SWARM_BRAIN_SLEEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Coordination strength modulation by sleep state */
#define SWARM_BRAIN_SLEEP_COORD_AWAKE         1.0f
#define SWARM_BRAIN_SLEEP_COORD_DROWSY        0.5f
#define SWARM_BRAIN_SLEEP_COORD_LIGHT_NREM    0.2f
#define SWARM_BRAIN_SLEEP_COORD_DEEP_NREM     0.05f
#define SWARM_BRAIN_SLEEP_COORD_REM           0.3f

/* Heartbeat interval multipliers by sleep state */
#define SWARM_BRAIN_SLEEP_HB_AWAKE            1.0f
#define SWARM_BRAIN_SLEEP_HB_DROWSY           2.0f
#define SWARM_BRAIN_SLEEP_HB_LIGHT_NREM       5.0f
#define SWARM_BRAIN_SLEEP_HB_DEEP_NREM        10.0f
#define SWARM_BRAIN_SLEEP_HB_REM              3.0f

typedef struct {
    bool enable_coord_modulation;
    bool enable_heartbeat_modulation;
    bool enable_coherence_modulation;
    float modulation_strength;
} swarm_brain_sleep_config_t;

typedef struct {
    float coordination_factor;
    float heartbeat_multiplier;
    float coherence_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool coordination_enabled;
} swarm_brain_sleep_effects_t;

typedef struct swarm_brain_sleep_bridge_struct* swarm_brain_sleep_bridge_t;

int swarm_brain_sleep_default_config(swarm_brain_sleep_config_t* config);
swarm_brain_sleep_bridge_t swarm_brain_sleep_bridge_create(
    const swarm_brain_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_brain_sleep_bridge_destroy(swarm_brain_sleep_bridge_t bridge);
int swarm_brain_sleep_update(swarm_brain_sleep_bridge_t bridge);
int swarm_brain_sleep_get_effects(const swarm_brain_sleep_bridge_t bridge,
                                   swarm_brain_sleep_effects_t* effects);
float swarm_brain_sleep_get_coordination(const swarm_brain_sleep_bridge_t bridge, float base);
uint32_t swarm_brain_sleep_get_heartbeat_interval(const swarm_brain_sleep_bridge_t bridge, uint32_t base_ms);
float swarm_brain_sleep_get_coherence(const swarm_brain_sleep_bridge_t bridge, float base);

float swarm_brain_sleep_get_coord_factor(sleep_state_t state);
float swarm_brain_sleep_get_heartbeat_factor(sleep_state_t state);
float swarm_brain_sleep_get_coherence_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_BRAIN_SLEEP_BRIDGE_H */
