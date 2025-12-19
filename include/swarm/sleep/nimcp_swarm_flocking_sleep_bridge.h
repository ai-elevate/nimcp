/**
 * @file nimcp_swarm_flocking_sleep_bridge.h
 * @brief Sleep-Swarm Flocking Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm flocking behaviors
 * WHY:  Sleep states affect flocking forces (separation, alignment, cohesion)
 * HOW:  Sleep state modulates flocking weights, update frequency, and formation tightness
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full flocking (tight formation)
 * - DROWSY: Loose flocking (relaxed formation)
 * - LIGHT_NREM: Minimal flocking (isolated resting)
 * - DEEP_NREM: No flocking (stationary)
 * - REM: Sporadic flocking (occasional movement bursts)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_FLOCKING_SLEEP_BRIDGE_H
#define NIMCP_SWARM_FLOCKING_SLEEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flocking force modulation by sleep state */
#define SWARM_FLOCKING_SLEEP_FORCE_AWAKE         1.0f
#define SWARM_FLOCKING_SLEEP_FORCE_DROWSY        0.6f
#define SWARM_FLOCKING_SLEEP_FORCE_LIGHT_NREM    0.2f
#define SWARM_FLOCKING_SLEEP_FORCE_DEEP_NREM     0.0f
#define SWARM_FLOCKING_SLEEP_FORCE_REM           0.3f

/* Update frequency modulation by sleep state */
#define SWARM_FLOCKING_SLEEP_UPDATE_AWAKE        1.0f
#define SWARM_FLOCKING_SLEEP_UPDATE_DROWSY       0.5f
#define SWARM_FLOCKING_SLEEP_UPDATE_LIGHT_NREM   0.2f
#define SWARM_FLOCKING_SLEEP_UPDATE_DEEP_NREM    0.05f
#define SWARM_FLOCKING_SLEEP_UPDATE_REM          0.3f

typedef struct {
    bool enable_force_modulation;
    bool enable_update_modulation;
    bool enable_formation_modulation;
    float modulation_strength;
} swarm_flocking_sleep_config_t;

typedef struct {
    float separation_factor;
    float alignment_factor;
    float cohesion_factor;
    float update_frequency_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool flocking_enabled;
} swarm_flocking_sleep_effects_t;

typedef struct swarm_flocking_sleep_bridge_struct* swarm_flocking_sleep_bridge_t;

int swarm_flocking_sleep_default_config(swarm_flocking_sleep_config_t* config);
swarm_flocking_sleep_bridge_t swarm_flocking_sleep_bridge_create(
    const swarm_flocking_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_flocking_sleep_bridge_destroy(swarm_flocking_sleep_bridge_t bridge);
int swarm_flocking_sleep_update(swarm_flocking_sleep_bridge_t bridge);
int swarm_flocking_sleep_get_effects(const swarm_flocking_sleep_bridge_t bridge,
                                      swarm_flocking_sleep_effects_t* effects);
float swarm_flocking_sleep_get_separation(const swarm_flocking_sleep_bridge_t bridge, float base);
float swarm_flocking_sleep_get_alignment(const swarm_flocking_sleep_bridge_t bridge, float base);
float swarm_flocking_sleep_get_cohesion(const swarm_flocking_sleep_bridge_t bridge, float base);

float swarm_flocking_sleep_get_force_factor(sleep_state_t state);
float swarm_flocking_sleep_get_update_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_FLOCKING_SLEEP_BRIDGE_H */
