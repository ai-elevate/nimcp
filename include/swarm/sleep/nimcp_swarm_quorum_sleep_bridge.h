/**
 * @file nimcp_swarm_quorum_sleep_bridge.h
 * @brief Sleep-Swarm Quorum Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm quorum sensing
 * WHY:  Sleep states affect quorum decision thresholds and commitment rates
 * HOW:  Sleep state modulates decision threshold, signal decay, and commitment rate
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Normal quorum sensing (standard thresholds)
 * - DROWSY: Higher thresholds (more conservative)
 * - LIGHT_NREM: Minimal quorum sensing
 * - DEEP_NREM: No quorum decisions (offline)
 * - REM: Exploratory decisions (lower thresholds)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_QUORUM_SLEEP_BRIDGE_H
#define NIMCP_SWARM_QUORUM_SLEEP_BRIDGE_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decision threshold modulation by sleep state (higher = stricter) */
#define SWARM_QUORUM_SLEEP_THRESH_AWAKE         1.0f
#define SWARM_QUORUM_SLEEP_THRESH_DROWSY        1.2f
#define SWARM_QUORUM_SLEEP_THRESH_LIGHT_NREM    1.4f
#define SWARM_QUORUM_SLEEP_THRESH_DEEP_NREM     1.6f
#define SWARM_QUORUM_SLEEP_THRESH_REM           0.9f

/* Signal decay modulation by sleep state (lower = slower decay) */
#define SWARM_QUORUM_SLEEP_DECAY_AWAKE          1.0f
#define SWARM_QUORUM_SLEEP_DECAY_DROWSY         0.8f
#define SWARM_QUORUM_SLEEP_DECAY_LIGHT_NREM     0.6f
#define SWARM_QUORUM_SLEEP_DECAY_DEEP_NREM      0.5f
#define SWARM_QUORUM_SLEEP_DECAY_REM            0.7f

/* Commitment rate modulation by sleep state */
#define SWARM_QUORUM_SLEEP_COMMIT_AWAKE         1.0f
#define SWARM_QUORUM_SLEEP_COMMIT_DROWSY        0.7f
#define SWARM_QUORUM_SLEEP_COMMIT_LIGHT_NREM    0.4f
#define SWARM_QUORUM_SLEEP_COMMIT_DEEP_NREM     0.2f
#define SWARM_QUORUM_SLEEP_COMMIT_REM           0.5f

typedef struct {
    bool enable_threshold_modulation;
    bool enable_decay_modulation;
    bool enable_commitment_modulation;
    float modulation_strength;
} swarm_quorum_sleep_config_t;

typedef struct {
    float decision_threshold_factor;
    float signal_decay_factor;
    float commitment_rate_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool quorum_enabled;
} swarm_quorum_sleep_effects_t;

typedef struct swarm_quorum_sleep_bridge_struct* swarm_quorum_sleep_bridge_t;

int swarm_quorum_sleep_default_config(swarm_quorum_sleep_config_t* config);
swarm_quorum_sleep_bridge_t swarm_quorum_sleep_bridge_create(
    const swarm_quorum_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_quorum_sleep_bridge_destroy(swarm_quorum_sleep_bridge_t bridge);
int swarm_quorum_sleep_update(swarm_quorum_sleep_bridge_t bridge);
int swarm_quorum_sleep_get_effects(const swarm_quorum_sleep_bridge_t bridge,
                                    swarm_quorum_sleep_effects_t* effects);
float swarm_quorum_sleep_get_decision_threshold(const swarm_quorum_sleep_bridge_t bridge, float base);
float swarm_quorum_sleep_get_signal_decay(const swarm_quorum_sleep_bridge_t bridge, float base);
float swarm_quorum_sleep_get_commitment_rate(const swarm_quorum_sleep_bridge_t bridge, float base);

float swarm_quorum_sleep_get_thresh_factor(sleep_state_t state);
float swarm_quorum_sleep_get_decay_factor(sleep_state_t state);
float swarm_quorum_sleep_get_commit_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_QUORUM_SLEEP_BRIDGE_H */
