/**
 * @file nimcp_swarm_consciousness_sleep_bridge.h
 * @brief Sleep-Swarm Consciousness Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm collective consciousness
 * WHY:  Sleep states fundamentally alter collective phi and consciousness emergence
 * HOW:  Sleep state modulates phi aggregation, network integration, and consciousness thresholds
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full collective consciousness (integrated phi)
 * - DROWSY: Reduced consciousness (fragmented awareness)
 * - LIGHT_NREM: Minimal consciousness (isolated processing)
 * - DEEP_NREM: No collective consciousness (dormant state)
 * - REM: Sporadic consciousness (dream-like collective states)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_CONSCIOUSNESS_SLEEP_BRIDGE_H
#define NIMCP_SWARM_CONSCIOUSNESS_SLEEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Phi aggregation modulation by sleep state */
#define SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE         1.0f
#define SWARM_CONSCIOUSNESS_SLEEP_PHI_DROWSY        0.5f
#define SWARM_CONSCIOUSNESS_SLEEP_PHI_LIGHT_NREM    0.2f
#define SWARM_CONSCIOUSNESS_SLEEP_PHI_DEEP_NREM     0.05f
#define SWARM_CONSCIOUSNESS_SLEEP_PHI_REM           0.3f

/* Network integration modulation by sleep state */
#define SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE         1.0f
#define SWARM_CONSCIOUSNESS_SLEEP_INT_DROWSY        0.5f
#define SWARM_CONSCIOUSNESS_SLEEP_INT_LIGHT_NREM    0.2f
#define SWARM_CONSCIOUSNESS_SLEEP_INT_DEEP_NREM     0.05f
#define SWARM_CONSCIOUSNESS_SLEEP_INT_REM           0.3f

typedef struct {
    bool enable_phi_modulation;
    bool enable_integration_modulation;
    bool enable_coherence_modulation;
    float modulation_strength;
} swarm_consciousness_sleep_config_t;

typedef struct {
    float phi_factor;
    float integration_factor;
    float coherence_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool consciousness_enabled;
} swarm_consciousness_sleep_effects_t;

typedef struct swarm_consciousness_sleep_bridge_struct* swarm_consciousness_sleep_bridge_t;

int swarm_consciousness_sleep_default_config(swarm_consciousness_sleep_config_t* config);
swarm_consciousness_sleep_bridge_t swarm_consciousness_sleep_bridge_create(
    const swarm_consciousness_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_consciousness_sleep_bridge_destroy(swarm_consciousness_sleep_bridge_t bridge);
int swarm_consciousness_sleep_update(swarm_consciousness_sleep_bridge_t bridge);
int swarm_consciousness_sleep_get_effects(const swarm_consciousness_sleep_bridge_t bridge,
                                           swarm_consciousness_sleep_effects_t* effects);
float swarm_consciousness_sleep_get_phi(const swarm_consciousness_sleep_bridge_t bridge, float base);
float swarm_consciousness_sleep_get_integration(const swarm_consciousness_sleep_bridge_t bridge, float base);
float swarm_consciousness_sleep_get_coherence(const swarm_consciousness_sleep_bridge_t bridge, float base);

float swarm_consciousness_sleep_get_phi_factor(sleep_state_t state);
float swarm_consciousness_sleep_get_integration_factor(sleep_state_t state);
float swarm_consciousness_sleep_get_coherence_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_CONSCIOUSNESS_SLEEP_BRIDGE_H */
