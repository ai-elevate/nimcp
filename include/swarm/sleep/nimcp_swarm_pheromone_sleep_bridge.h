/**
 * @file nimcp_swarm_pheromone_sleep_bridge.h
 * @brief Sleep-Swarm Pheromone Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm pheromone signaling
 * WHY:  Sleep states affect pheromone production, decay, and detection sensitivity
 * HOW:  Sleep state modulates decay rate, diffusion rate, and detection threshold
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full pheromone signaling (active communication)
 * - DROWSY: Reduced signaling (preparing for rest)
 * - LIGHT_NREM: Minimal signaling, slower decay
 * - DEEP_NREM: No signaling, slowest decay
 * - REM: Sporadic signaling (internal simulation)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_PHEROMONE_SLEEP_BRIDGE_H
#define NIMCP_SWARM_PHEROMONE_SLEEP_BRIDGE_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decay rate modulation by sleep state (lower = slower decay) */
#define SWARM_PHEROMONE_SLEEP_DECAY_AWAKE         1.0f
#define SWARM_PHEROMONE_SLEEP_DECAY_DROWSY        0.8f
#define SWARM_PHEROMONE_SLEEP_DECAY_LIGHT_NREM    0.6f
#define SWARM_PHEROMONE_SLEEP_DECAY_DEEP_NREM     0.5f
#define SWARM_PHEROMONE_SLEEP_DECAY_REM           0.7f

/* Diffusion rate modulation by sleep state */
#define SWARM_PHEROMONE_SLEEP_DIFF_AWAKE          1.0f
#define SWARM_PHEROMONE_SLEEP_DIFF_DROWSY         0.8f
#define SWARM_PHEROMONE_SLEEP_DIFF_LIGHT_NREM     0.6f
#define SWARM_PHEROMONE_SLEEP_DIFF_DEEP_NREM      0.5f
#define SWARM_PHEROMONE_SLEEP_DIFF_REM            0.7f

/* Detection threshold modulation by sleep state (higher = less sensitive) */
#define SWARM_PHEROMONE_SLEEP_DETECT_AWAKE        1.0f
#define SWARM_PHEROMONE_SLEEP_DETECT_DROWSY       1.1f
#define SWARM_PHEROMONE_SLEEP_DETECT_LIGHT_NREM   1.3f
#define SWARM_PHEROMONE_SLEEP_DETECT_DEEP_NREM    1.4f
#define SWARM_PHEROMONE_SLEEP_DETECT_REM          1.2f

typedef struct {
    bool enable_decay_modulation;
    bool enable_diffusion_modulation;
    bool enable_detection_modulation;
    float modulation_strength;
} swarm_pheromone_sleep_config_t;

typedef struct {
    float decay_rate_factor;
    float diffusion_rate_factor;
    float detection_threshold_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool signaling_enabled;
} swarm_pheromone_sleep_effects_t;

typedef struct swarm_pheromone_sleep_bridge_struct* swarm_pheromone_sleep_bridge_t;

int swarm_pheromone_sleep_default_config(swarm_pheromone_sleep_config_t* config);
swarm_pheromone_sleep_bridge_t swarm_pheromone_sleep_bridge_create(
    const swarm_pheromone_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_pheromone_sleep_bridge_destroy(swarm_pheromone_sleep_bridge_t bridge);
int swarm_pheromone_sleep_update(swarm_pheromone_sleep_bridge_t bridge);
int swarm_pheromone_sleep_get_effects(const swarm_pheromone_sleep_bridge_t bridge,
                                       swarm_pheromone_sleep_effects_t* effects);
float swarm_pheromone_sleep_get_decay_rate(const swarm_pheromone_sleep_bridge_t bridge, float base);
float swarm_pheromone_sleep_get_diffusion_rate(const swarm_pheromone_sleep_bridge_t bridge, float base);
float swarm_pheromone_sleep_get_detection_threshold(const swarm_pheromone_sleep_bridge_t bridge, float base);

float swarm_pheromone_sleep_get_decay_factor(sleep_state_t state);
float swarm_pheromone_sleep_get_diff_factor(sleep_state_t state);
float swarm_pheromone_sleep_get_detect_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_PHEROMONE_SLEEP_BRIDGE_H */
