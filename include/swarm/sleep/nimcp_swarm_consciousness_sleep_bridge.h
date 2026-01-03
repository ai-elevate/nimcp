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
#include "utils/bridge/nimcp_bridge_base.h"
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

/*============================================================================
 * Bidirectional Integration: Consciousness → Sleep
 *============================================================================
 * BIOLOGICAL BASIS:
 * - High consciousness (transcendent state) promotes wakefulness
 * - Low consciousness increases sleep pressure/need
 * - Consciousness quality affects circadian rhythm modulation
 * - Collective phi affects sleep synchronization across swarm
 *============================================================================*/

/**
 * @brief Consciousness-based sleep modulation factors
 *
 * WHAT: How consciousness states affect sleep drive
 * WHY:  High consciousness should promote wakefulness; low promotes sleep
 * HOW:  Modulation factors applied to sleep pressure calculations
 */
#define SWARM_CONSCIOUSNESS_TO_SLEEP_DORMANT       1.5f   /* Increases sleep pressure */
#define SWARM_CONSCIOUSNESS_TO_SLEEP_EMERGING      1.0f   /* Neutral */
#define SWARM_CONSCIOUSNESS_TO_SLEEP_UNIFIED       0.7f   /* Reduces sleep pressure */
#define SWARM_CONSCIOUSNESS_TO_SLEEP_TRANSCENDENT  0.3f   /* Strongly promotes wakefulness */

/**
 * @brief Modulation applied to sleep system from consciousness
 */
typedef struct {
    float sleep_pressure_modifier;     /**< Multiplier for sleep pressure */
    float wakefulness_boost;           /**< Additional wakefulness factor [0-1] */
    float circadian_phase_shift;       /**< Phase shift in hours [-2, +2] */
    bool suppress_sleep_transition;    /**< Block transition to sleep */
    uint32_t consciousness_state;      /**< Current swarm consciousness state */
    float collective_phi;              /**< Current collective phi */
} swarm_sleep_consciousness_modulation_t;

/**
 * @brief Connect consciousness context to sleep bridge for bidirectional updates
 *
 * @param bridge Sleep bridge
 * @param consciousness_ctx Swarm consciousness context
 * @return 0 on success, -1 on error
 */
struct swarm_consciousness_ctx;
int swarm_consciousness_sleep_connect_consciousness(
    swarm_consciousness_sleep_bridge_t bridge,
    struct swarm_consciousness_ctx* consciousness_ctx);

/**
 * @brief Disconnect consciousness from sleep bridge
 *
 * @param bridge Sleep bridge
 */
void swarm_consciousness_sleep_disconnect_consciousness(
    swarm_consciousness_sleep_bridge_t bridge);

/**
 * @brief Update sleep pressure based on consciousness state
 *
 * Called when consciousness state changes to modulate sleep system.
 *
 * @param bridge Sleep bridge
 * @param consciousness_state Current swarm consciousness state (swarm_consciousness_state_t)
 * @param collective_phi Current collective phi value
 * @return 0 on success, -1 on error
 */
int swarm_consciousness_sleep_on_consciousness_change(
    swarm_consciousness_sleep_bridge_t bridge,
    uint32_t consciousness_state,
    float collective_phi);

/**
 * @brief Get consciousness-based sleep modulation
 *
 * @param bridge Sleep bridge
 * @param modulation Output modulation values
 * @return 0 on success, -1 on error
 */
int swarm_consciousness_sleep_get_consciousness_modulation(
    const swarm_consciousness_sleep_bridge_t bridge,
    swarm_sleep_consciousness_modulation_t* modulation);

/**
 * @brief Compute sleep pressure modifier from consciousness state
 *
 * @param consciousness_state Swarm consciousness state
 * @return Sleep pressure modifier (< 1 reduces pressure, > 1 increases)
 */
float swarm_consciousness_sleep_get_pressure_modifier(uint32_t consciousness_state);

/**
 * @brief Check if consciousness state blocks sleep transition
 *
 * Transcendent consciousness may block transition to deep sleep.
 *
 * @param bridge Sleep bridge
 * @return true if sleep transition should be blocked
 */
bool swarm_consciousness_sleep_blocks_transition(
    const swarm_consciousness_sleep_bridge_t bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_CONSCIOUSNESS_SLEEP_BRIDGE_H */
