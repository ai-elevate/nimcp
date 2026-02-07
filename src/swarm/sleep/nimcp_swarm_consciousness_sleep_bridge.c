/**
 * @file nimcp_swarm_consciousness_sleep_bridge.c
 * @brief Swarm Consciousness-Sleep Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm consciousness based on sleep state
 * WHY:  Collective phi and integration should vary with consciousness level
 * HOW:  Sleep state callbacks dynamically adjust consciousness parameters
 *
 * @author NIMCP Development Team
 */

#include "swarm/sleep/nimcp_swarm_consciousness_sleep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(swarm_consciousness_sleep_bridge)

struct swarm_consciousness_sleep_bridge_struct {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    swarm_consciousness_sleep_config_t config;
    sleep_system_t sleep_system;
    swarm_consciousness_sleep_effects_t effects;
    bool callback_registered;

    /* Bidirectional: Consciousness → Sleep */
    struct swarm_consciousness_ctx* consciousness_ctx;
    swarm_sleep_consciousness_modulation_t consciousness_modulation;
};

static void swarm_consciousness_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    swarm_consciousness_sleep_bridge_t bridge = (swarm_consciousness_sleep_bridge_t)user_data;
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.current_state = new_state;
    bridge->effects.phi_factor = swarm_consciousness_sleep_get_phi_factor(new_state);
    bridge->effects.integration_factor = swarm_consciousness_sleep_get_integration_factor(new_state);
    bridge->effects.coherence_factor = swarm_consciousness_sleep_get_coherence_factor(new_state);
    bridge->effects.consciousness_enabled = (new_state != SLEEP_STATE_DEEP_NREM);

    NIMCP_LOGGING_DEBUG("Swarm consciousness sleep state changed to %d, phi=%.2f, integration=%.2f",
                        new_state, bridge->effects.phi_factor, bridge->effects.integration_factor);

    nimcp_mutex_unlock(bridge->base.mutex);
}

int swarm_consciousness_sleep_default_config(swarm_consciousness_sleep_config_t* config)
{
    if (!config) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;

    }

    config->enable_phi_modulation = true;
    config->enable_integration_modulation = true;
    config->enable_coherence_modulation = true;
    config->modulation_strength = 1.0f;

    return 0;
}

swarm_consciousness_sleep_bridge_t swarm_consciousness_sleep_bridge_create(
    const swarm_consciousness_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!config || !sleep_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm consciousness sleep bridge creation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "swarm_consciousness_sleep_bridge_create: required parameter is NULL (config, sleep_system)");
        return NULL;
    }

    swarm_consciousness_sleep_bridge_t bridge =
        (swarm_consciousness_sleep_bridge_t)nimcp_malloc(sizeof(struct swarm_consciousness_sleep_bridge_struct));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm consciousness sleep bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(struct swarm_consciousness_sleep_bridge_struct));
    memcpy(&bridge->config, config, sizeof(swarm_consciousness_sleep_config_t));
    bridge->sleep_system = sleep_system;

    if (bridge_base_init(&bridge->base, 0, "swarm_consciousness_sleep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm consciousness sleep bridge");
        nimcp_free(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "swarm_consciousness_sleep_bridge_create: bridge->base is NULL");
        return NULL;
    }

    bridge->effects.phi_factor = SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE;
    bridge->effects.integration_factor = SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE;
    bridge->effects.coherence_factor = 1.0f;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.consciousness_enabled = true;

    bridge->callback_registered = sleep_register_state_callback(
        sleep_system, swarm_consciousness_on_sleep_state_change, bridge);

    if (bridge->callback_registered) {
        sleep_state_t initial = sleep_get_current_state(sleep_system);
        swarm_consciousness_on_sleep_state_change(initial, bridge);
    }

    NIMCP_LOGGING_INFO("Swarm consciousness sleep bridge created");
    return bridge;
}

void swarm_consciousness_sleep_bridge_destroy(swarm_consciousness_sleep_bridge_t bridge)
{
    if (!bridge) return;

    if (bridge->callback_registered && bridge->sleep_system) {
        sleep_unregister_state_callback(bridge->sleep_system,
                                        swarm_consciousness_on_sleep_state_change, bridge);
    }

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm consciousness sleep bridge destroyed");
}

int swarm_consciousness_sleep_update(swarm_consciousness_sleep_bridge_t bridge)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->sleep_system) {
        bridge->effects.sleep_pressure = sleep_get_pressure(bridge->sleep_system);
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int swarm_consciousness_sleep_get_effects(const swarm_consciousness_sleep_bridge_t bridge,
                                           swarm_consciousness_sleep_effects_t* effects)
{
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "swarm_consciousness_sleep_get_effects: required parameter is NULL (bridge, effects)");
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(effects, &bridge->effects, sizeof(swarm_consciousness_sleep_effects_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_consciousness_sleep_get_phi(const swarm_consciousness_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_phi_modulation ?
        bridge->effects.phi_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_consciousness_sleep_get_integration(const swarm_consciousness_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_integration_modulation ?
        bridge->effects.integration_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_consciousness_sleep_get_coherence(const swarm_consciousness_sleep_bridge_t bridge, float base)
{
    if (!bridge) return base;

    nimcp_mutex_lock(bridge->base.mutex);
    float factor = bridge->config.enable_coherence_modulation ?
        bridge->effects.coherence_factor : 1.0f;
    float result = base * (1.0f + (factor - 1.0f) * bridge->config.modulation_strength);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

float swarm_consciousness_sleep_get_phi_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_CONSCIOUSNESS_SLEEP_PHI_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_CONSCIOUSNESS_SLEEP_PHI_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_CONSCIOUSNESS_SLEEP_PHI_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_CONSCIOUSNESS_SLEEP_PHI_REM;
        default:                     return SWARM_CONSCIOUSNESS_SLEEP_PHI_AWAKE;
    }
}

float swarm_consciousness_sleep_get_integration_factor(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE;
        case SLEEP_STATE_DROWSY:     return SWARM_CONSCIOUSNESS_SLEEP_INT_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return SWARM_CONSCIOUSNESS_SLEEP_INT_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return SWARM_CONSCIOUSNESS_SLEEP_INT_DEEP_NREM;
        case SLEEP_STATE_REM:        return SWARM_CONSCIOUSNESS_SLEEP_INT_REM;
        default:                     return SWARM_CONSCIOUSNESS_SLEEP_INT_AWAKE;
    }
}

float swarm_consciousness_sleep_get_coherence_factor(sleep_state_t state)
{
    /* Coherence uses same pattern as phi */
    return swarm_consciousness_sleep_get_phi_factor(state);
}

/*============================================================================
 * Bidirectional Integration: Consciousness → Sleep
 *============================================================================*/

int swarm_consciousness_sleep_connect_consciousness(
    swarm_consciousness_sleep_bridge_t bridge,
    struct swarm_consciousness_ctx* consciousness_ctx)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->consciousness_ctx = consciousness_ctx;

    /* Initialize modulation to neutral */
    bridge->consciousness_modulation.sleep_pressure_modifier = 1.0f;
    bridge->consciousness_modulation.wakefulness_boost = 0.0f;
    bridge->consciousness_modulation.circadian_phase_shift = 0.0f;
    bridge->consciousness_modulation.suppress_sleep_transition = false;
    bridge->consciousness_modulation.consciousness_state = 0;  /* DORMANT */
    bridge->consciousness_modulation.collective_phi = 0.0f;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected consciousness context to sleep bridge");
    return 0;
}

void swarm_consciousness_sleep_disconnect_consciousness(
    swarm_consciousness_sleep_bridge_t bridge)
{
    if (!bridge) return;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->consciousness_ctx = NULL;

    /* Reset modulation to neutral */
    bridge->consciousness_modulation.sleep_pressure_modifier = 1.0f;
    bridge->consciousness_modulation.wakefulness_boost = 0.0f;
    bridge->consciousness_modulation.suppress_sleep_transition = false;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected consciousness context from sleep bridge");
}

int swarm_consciousness_sleep_on_consciousness_change(
    swarm_consciousness_sleep_bridge_t bridge,
    uint32_t consciousness_state,
    float collective_phi)
{
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Store state */
    bridge->consciousness_modulation.consciousness_state = consciousness_state;
    bridge->consciousness_modulation.collective_phi = collective_phi;

    /* Compute pressure modifier based on consciousness state */
    bridge->consciousness_modulation.sleep_pressure_modifier =
        swarm_consciousness_sleep_get_pressure_modifier(consciousness_state);

    /* Compute wakefulness boost from phi */
    /* Higher phi = more wakeful, range [0, 1] */
    float normalized_phi = collective_phi / 10.0f;  /* Assume max phi ~10 */
    if (normalized_phi > 1.0f) normalized_phi = 1.0f;
    bridge->consciousness_modulation.wakefulness_boost = normalized_phi * 0.5f;

    /* Transcendent state blocks sleep transition */
    bridge->consciousness_modulation.suppress_sleep_transition =
        (consciousness_state == 3);  /* SWARM_CONSCIOUSNESS_TRANSCENDENT */

    /* Apply modulation to sleep system if connected */
    if (bridge->sleep_system) {
        /* Modulate sleep pressure */
        float current_pressure = sleep_get_pressure(bridge->sleep_system);
        float modified_pressure = current_pressure *
            bridge->consciousness_modulation.sleep_pressure_modifier;

        /* Note: Would call sleep_set_pressure_modifier() if it exists */
        NIMCP_LOGGING_DEBUG("Consciousness modulating sleep: state=%u, phi=%.2f, pressure_mod=%.2f",
                            consciousness_state, collective_phi,
                            bridge->consciousness_modulation.sleep_pressure_modifier);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int swarm_consciousness_sleep_get_consciousness_modulation(
    const swarm_consciousness_sleep_bridge_t bridge,
    swarm_sleep_consciousness_modulation_t* modulation)
{
    if (!bridge || !modulation) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "swarm_consciousness_sleep_get_consciousness_modulation: required parameter is NULL (bridge, modulation)");
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    memcpy(modulation, &bridge->consciousness_modulation,
           sizeof(swarm_sleep_consciousness_modulation_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float swarm_consciousness_sleep_get_pressure_modifier(uint32_t consciousness_state)
{
    switch (consciousness_state) {
        case 0:  /* SWARM_CONSCIOUSNESS_DORMANT */
            return SWARM_CONSCIOUSNESS_TO_SLEEP_DORMANT;
        case 1:  /* SWARM_CONSCIOUSNESS_EMERGING */
            return SWARM_CONSCIOUSNESS_TO_SLEEP_EMERGING;
        case 2:  /* SWARM_CONSCIOUSNESS_UNIFIED */
            return SWARM_CONSCIOUSNESS_TO_SLEEP_UNIFIED;
        case 3:  /* SWARM_CONSCIOUSNESS_TRANSCENDENT */
            return SWARM_CONSCIOUSNESS_TO_SLEEP_TRANSCENDENT;
        default:
            return 1.0f;  /* Neutral */
    }
}

bool swarm_consciousness_sleep_blocks_transition(
    const swarm_consciousness_sleep_bridge_t bridge)
{
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "swarm_consciousness_sleep_blocks_transition: bridge is NULL");
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    bool blocks = bridge->consciousness_modulation.suppress_sleep_transition;
    nimcp_mutex_unlock(bridge->base.mutex);

    return blocks;
}
