/**
 * @file nimcp_swarm_immune_sleep_bridge.h
 * @brief Sleep-Swarm Immune Integration Bridge
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Bidirectional integration between sleep/wake system and swarm immune system
 * WHY:  Sleep states fundamentally alter immune responsiveness and threat detection
 * HOW:  Sleep state modulates threat detection sensitivity, response intensity, memory consolidation
 *
 * BIOLOGICAL BASIS:
 * - AWAKE: Full threat detection sensitivity
 * - DROWSY: Slightly reduced sensitivity (transition state)
 * - LIGHT_NREM: Reduced sensitivity, enhanced memory consolidation
 * - DEEP_NREM: Minimal detection (only critical threats), maximum consolidation
 * - REM: Moderate sensitivity (threat simulation and learning)
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_SWARM_IMMUNE_SLEEP_BRIDGE_H
#define NIMCP_SWARM_IMMUNE_SLEEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "cognitive/nimcp_sleep_wake.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Detection sensitivity modulation by sleep state */
#define SWARM_IMMUNE_SLEEP_DETECT_AWAKE         1.0f
#define SWARM_IMMUNE_SLEEP_DETECT_DROWSY        0.8f
#define SWARM_IMMUNE_SLEEP_DETECT_LIGHT_NREM    0.5f
#define SWARM_IMMUNE_SLEEP_DETECT_DEEP_NREM     0.2f
#define SWARM_IMMUNE_SLEEP_DETECT_REM           0.6f

/* Response intensity modulation by sleep state */
#define SWARM_IMMUNE_SLEEP_RESPONSE_AWAKE       1.0f
#define SWARM_IMMUNE_SLEEP_RESPONSE_DROWSY      0.8f
#define SWARM_IMMUNE_SLEEP_RESPONSE_LIGHT_NREM  0.6f
#define SWARM_IMMUNE_SLEEP_RESPONSE_DEEP_NREM   0.3f
#define SWARM_IMMUNE_SLEEP_RESPONSE_REM         0.7f

/* Memory consolidation boost by sleep state */
#define SWARM_IMMUNE_SLEEP_MEMORY_AWAKE         1.0f
#define SWARM_IMMUNE_SLEEP_MEMORY_DROWSY        0.9f
#define SWARM_IMMUNE_SLEEP_MEMORY_LIGHT_NREM    1.2f
#define SWARM_IMMUNE_SLEEP_MEMORY_DEEP_NREM     1.5f
#define SWARM_IMMUNE_SLEEP_MEMORY_REM           1.3f

typedef struct {
    bool enable_detection_modulation;
    bool enable_response_modulation;
    bool enable_memory_consolidation;
    float modulation_strength;
} swarm_immune_sleep_config_t;

typedef struct {
    float detection_sensitivity;
    float response_intensity;
    float memory_consolidation;
    sleep_state_t current_state;
    float sleep_pressure;
    bool suppress_non_critical;
} swarm_immune_sleep_effects_t;

typedef struct swarm_immune_sleep_bridge_struct* swarm_immune_sleep_bridge_t;

int swarm_immune_sleep_default_config(swarm_immune_sleep_config_t* config);
swarm_immune_sleep_bridge_t swarm_immune_sleep_bridge_create(
    const swarm_immune_sleep_config_t* config,
    sleep_system_t sleep_system);
void swarm_immune_sleep_bridge_destroy(swarm_immune_sleep_bridge_t bridge);
int swarm_immune_sleep_update(swarm_immune_sleep_bridge_t bridge);
int swarm_immune_sleep_get_effects(const swarm_immune_sleep_bridge_t bridge,
                                    swarm_immune_sleep_effects_t* effects);
float swarm_immune_sleep_get_detection_sensitivity(const swarm_immune_sleep_bridge_t bridge, float base);
float swarm_immune_sleep_get_response_intensity(const swarm_immune_sleep_bridge_t bridge, float base);
float swarm_immune_sleep_get_memory_boost(const swarm_immune_sleep_bridge_t bridge, float base);

float swarm_immune_sleep_get_detect_factor(sleep_state_t state);
float swarm_immune_sleep_get_response_factor(sleep_state_t state);
float swarm_immune_sleep_get_memory_factor(sleep_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_IMMUNE_SLEEP_BRIDGE_H */
