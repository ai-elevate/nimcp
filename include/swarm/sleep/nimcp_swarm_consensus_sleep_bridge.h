/**
 * @file nimcp_swarm_consensus_sleep_bridge.h
 * @brief Sleep-Swarm Consensus Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm consensus voting
 * WHY:  Sleep states affect collective decision-making speed and quorum requirements
 * HOW:  Sleep state modulates voting frequency, quorum threshold, and agreement requirements
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full voting (normal consensus)
 * - DROWSY: Slower voting, higher quorum threshold
 * - LIGHT_NREM: Minimal voting (only critical decisions)
 * - DEEP_NREM: No voting (consensus offline)
 * - REM: Sporadic voting (exploratory decisions)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_CONSENSUS_SLEEP_BRIDGE_H
#define NIMCP_SWARM_CONSENSUS_SLEEP_BRIDGE_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Voting frequency modulation by sleep state */
#define SWARM_CONSENSUS_SLEEP_VOTE_AWAKE         1.0f
#define SWARM_CONSENSUS_SLEEP_VOTE_DROWSY        0.5f
#define SWARM_CONSENSUS_SLEEP_VOTE_LIGHT_NREM    0.1f
#define SWARM_CONSENSUS_SLEEP_VOTE_DEEP_NREM     0.0f
#define SWARM_CONSENSUS_SLEEP_VOTE_REM           0.2f

/* Quorum threshold modulation by sleep state (higher = stricter) */
#define SWARM_CONSENSUS_SLEEP_QUORUM_AWAKE       1.0f
#define SWARM_CONSENSUS_SLEEP_QUORUM_DROWSY      1.2f
#define SWARM_CONSENSUS_SLEEP_QUORUM_LIGHT_NREM  1.5f
#define SWARM_CONSENSUS_SLEEP_QUORUM_DEEP_NREM   2.0f
#define SWARM_CONSENSUS_SLEEP_QUORUM_REM         1.3f

typedef struct {
    bool enable_voting_modulation;
    bool enable_quorum_modulation;
    bool enable_timeout_modulation;
    float modulation_strength;
} swarm_consensus_sleep_config_t;

typedef struct {
    float voting_frequency_factor;
    float quorum_threshold_factor;
    float timeout_multiplier;
    sleep_state_t current_state;
    float sleep_pressure;
    bool consensus_enabled;
} swarm_consensus_sleep_effects_t;

typedef struct swarm_consensus_sleep_bridge_struct* swarm_consensus_sleep_bridge_t;

int swarm_consensus_sleep_default_config(swarm_consensus_sleep_config_t* config);
swarm_consensus_sleep_bridge_t swarm_consensus_sleep_bridge_create(
    const swarm_consensus_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_consensus_sleep_bridge_destroy(swarm_consensus_sleep_bridge_t bridge);
int swarm_consensus_sleep_update(swarm_consensus_sleep_bridge_t bridge);
int swarm_consensus_sleep_get_effects(const swarm_consensus_sleep_bridge_t bridge,
                                       swarm_consensus_sleep_effects_t* effects);
float swarm_consensus_sleep_get_voting_frequency(const swarm_consensus_sleep_bridge_t bridge, float base);
float swarm_consensus_sleep_get_quorum_threshold(const swarm_consensus_sleep_bridge_t bridge, float base);
uint32_t swarm_consensus_sleep_get_timeout(const swarm_consensus_sleep_bridge_t bridge, uint32_t base_ms);

float swarm_consensus_sleep_get_vote_factor(sleep_state_t state);
float swarm_consensus_sleep_get_quorum_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_CONSENSUS_SLEEP_BRIDGE_H */
