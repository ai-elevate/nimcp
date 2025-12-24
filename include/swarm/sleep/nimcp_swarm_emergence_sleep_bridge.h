/**
 * @file nimcp_swarm_emergence_sleep_bridge.h
 * @brief Sleep-Swarm Emergence Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm emergence tiers
 * WHY:  Sleep states affect collective intelligence emergence and capability unlocks
 * HOW:  Sleep state modulates tier transition thresholds and capability requirements
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full emergence capability (tier advancement)
 * - DROWSY: Reduced emergence (slower tier transitions)
 * - LIGHT_NREM: Tier consolidation (maintain current tier)
 * - DEEP_NREM: Emergence offline (no tier changes)
 * - REM: Exploratory emergence (capability testing)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_EMERGENCE_SLEEP_BRIDGE_H
#define NIMCP_SWARM_EMERGENCE_SLEEP_BRIDGE_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tier transition rate modulation by sleep state */
#define SWARM_EMERGENCE_SLEEP_TRANS_AWAKE         1.0f
#define SWARM_EMERGENCE_SLEEP_TRANS_DROWSY        0.5f
#define SWARM_EMERGENCE_SLEEP_TRANS_LIGHT_NREM    0.1f
#define SWARM_EMERGENCE_SLEEP_TRANS_DEEP_NREM     0.0f
#define SWARM_EMERGENCE_SLEEP_TRANS_REM           0.3f

/* Capability threshold modulation by sleep state */
#define SWARM_EMERGENCE_SLEEP_CAP_AWAKE           1.0f
#define SWARM_EMERGENCE_SLEEP_CAP_DROWSY          1.2f
#define SWARM_EMERGENCE_SLEEP_CAP_LIGHT_NREM      1.5f
#define SWARM_EMERGENCE_SLEEP_CAP_DEEP_NREM       2.0f
#define SWARM_EMERGENCE_SLEEP_CAP_REM             1.1f

typedef struct {
    bool enable_transition_modulation;
    bool enable_capability_modulation;
    bool enable_coherence_modulation;
    float modulation_strength;
} swarm_emergence_sleep_config_t;

typedef struct {
    float transition_rate_factor;
    float capability_threshold_factor;
    float coherence_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool emergence_enabled;
} swarm_emergence_sleep_effects_t;

typedef struct swarm_emergence_sleep_bridge_struct* swarm_emergence_sleep_bridge_t;

int swarm_emergence_sleep_default_config(swarm_emergence_sleep_config_t* config);
swarm_emergence_sleep_bridge_t swarm_emergence_sleep_bridge_create(
    const swarm_emergence_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_emergence_sleep_bridge_destroy(swarm_emergence_sleep_bridge_t bridge);
int swarm_emergence_sleep_update(swarm_emergence_sleep_bridge_t bridge);
int swarm_emergence_sleep_get_effects(const swarm_emergence_sleep_bridge_t bridge,
                                       swarm_emergence_sleep_effects_t* effects);
float swarm_emergence_sleep_get_transition_rate(const swarm_emergence_sleep_bridge_t bridge, float base);
float swarm_emergence_sleep_get_capability_threshold(const swarm_emergence_sleep_bridge_t bridge, float base);

float swarm_emergence_sleep_get_trans_factor(sleep_state_t state);
float swarm_emergence_sleep_get_cap_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_EMERGENCE_SLEEP_BRIDGE_H */
