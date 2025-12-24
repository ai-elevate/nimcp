/**
 * @file nimcp_swarm_signal_sleep_bridge.h
 * @brief Sleep-Swarm Signal Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm signal transmission
 * WHY:  Sleep states affect signal power, reception sensitivity, and latency tolerance
 * HOW:  Sleep state modulates transmission power, reception sensitivity, latency tolerance
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full signaling capacity (normal power)
 * - DROWSY: Reduced signaling (power conservation)
 * - LIGHT_NREM: Minimal signaling (essential only)
 * - DEEP_NREM: Emergency signaling only (lowest power)
 * - REM: Sporadic signaling (internal simulation)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_SIGNAL_SLEEP_BRIDGE_H
#define NIMCP_SWARM_SIGNAL_SLEEP_BRIDGE_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Transmission power modulation by sleep state */
#define SWARM_SIGNAL_SLEEP_POWER_AWAKE          1.0f
#define SWARM_SIGNAL_SLEEP_POWER_DROWSY         0.8f
#define SWARM_SIGNAL_SLEEP_POWER_LIGHT_NREM     0.6f
#define SWARM_SIGNAL_SLEEP_POWER_DEEP_NREM      0.4f
#define SWARM_SIGNAL_SLEEP_POWER_REM            0.7f

/* Reception sensitivity modulation by sleep state */
#define SWARM_SIGNAL_SLEEP_RECV_AWAKE           1.0f
#define SWARM_SIGNAL_SLEEP_RECV_DROWSY          0.7f
#define SWARM_SIGNAL_SLEEP_RECV_LIGHT_NREM      0.5f
#define SWARM_SIGNAL_SLEEP_RECV_DEEP_NREM       0.3f
#define SWARM_SIGNAL_SLEEP_RECV_REM             0.6f

/* Latency tolerance modulation by sleep state (higher = more tolerant) */
#define SWARM_SIGNAL_SLEEP_LATENCY_AWAKE        1.0f
#define SWARM_SIGNAL_SLEEP_LATENCY_DROWSY       1.2f
#define SWARM_SIGNAL_SLEEP_LATENCY_LIGHT_NREM   1.4f
#define SWARM_SIGNAL_SLEEP_LATENCY_DEEP_NREM    1.5f
#define SWARM_SIGNAL_SLEEP_LATENCY_REM          1.3f

typedef struct {
    bool enable_power_modulation;
    bool enable_reception_modulation;
    bool enable_latency_modulation;
    float modulation_strength;
} swarm_signal_sleep_config_t;

typedef struct {
    float transmission_power_factor;
    float reception_sensitivity_factor;
    float latency_tolerance_factor;
    sleep_state_t current_state;
    float sleep_pressure;
    bool signaling_enabled;
} swarm_signal_sleep_effects_t;

typedef struct swarm_signal_sleep_bridge_struct* swarm_signal_sleep_bridge_t;

int swarm_signal_sleep_default_config(swarm_signal_sleep_config_t* config);
swarm_signal_sleep_bridge_t swarm_signal_sleep_bridge_create(
    const swarm_signal_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_signal_sleep_bridge_destroy(swarm_signal_sleep_bridge_t bridge);
int swarm_signal_sleep_update(swarm_signal_sleep_bridge_t bridge);
int swarm_signal_sleep_get_effects(const swarm_signal_sleep_bridge_t bridge,
                                    swarm_signal_sleep_effects_t* effects);
float swarm_signal_sleep_get_transmission_power(const swarm_signal_sleep_bridge_t bridge, float base);
float swarm_signal_sleep_get_reception_sensitivity(const swarm_signal_sleep_bridge_t bridge, float base);
float swarm_signal_sleep_get_latency_tolerance(const swarm_signal_sleep_bridge_t bridge, float base);

float swarm_signal_sleep_get_power_factor(sleep_state_t state);
float swarm_signal_sleep_get_recv_factor(sleep_state_t state);
float swarm_signal_sleep_get_latency_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_SIGNAL_SLEEP_BRIDGE_H */
