/**
 * @file nimcp_stp_sleep_bridge.c
 * @brief Sleep-STP Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-17
 */

#include "plasticity/stp/nimcp_stp_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct stp_sleep_bridge_struct {
    stp_sleep_config_t config;
    sleep_system_t sleep_system;
    stp_sleep_effects_t effects;
    nimcp_platform_mutex_t* mutex;
    bool callback_registered;  /* Track if callback is registered for cleanup */
};

/* Forward declarations */
static void stp_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update STP parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Vesicle pools are replenished during sleep (synaptic homeostasis)
 * - Depression recovery is accelerated during NREM
 * - Release probability is reduced during deep sleep (conservation)
 * - REM shows enhanced release for memory replay
 */
static void stp_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    stp_sleep_bridge_t bridge = (stp_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("STP bridge received sleep state: %d", new_state);

    nimcp_platform_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    if (bridge->config.enable_u_modulation) {
        float u_base = stp_sleep_get_u_factor(new_state);
        bridge->effects.release_probability_factor =
            1.0f + (u_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_tau_d_modulation) {
        float tau_d_base = stp_sleep_get_tau_d_factor(new_state);
        bridge->effects.depression_recovery_factor =
            1.0f + (tau_d_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_tau_f_modulation) {
        float tau_f_base = stp_sleep_get_tau_f_factor(new_state);
        bridge->effects.facilitation_decay_factor =
            1.0f + (tau_f_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.vesicle_restoration_active = (new_state == SLEEP_STATE_DEEP_NREM);

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("STP modulated: U=%.2f, tau_D=%.2f, tau_F=%.2f",
                        bridge->effects.release_probability_factor,
                        bridge->effects.depression_recovery_factor,
                        bridge->effects.facilitation_decay_factor);
}

int stp_sleep_default_config(stp_sleep_config_t* config) {
    if (!config) return -1;
    config->enable_u_modulation = true;
    config->enable_tau_d_modulation = true;
    config->enable_tau_f_modulation = true;
    config->modulation_strength = 1.0f;
    return 0;
}

stp_sleep_bridge_t stp_sleep_bridge_create(
    const stp_sleep_config_t* config,
    sleep_system_t sleep_system)
{
    if (!sleep_system) {
        NIMCP_LOGGING_ERROR("stp_sleep_bridge_create: NULL sleep_system");
        return NULL;
    }

    struct stp_sleep_bridge_struct* bridge =
        (struct stp_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct stp_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct stp_sleep_bridge_struct));

    if (config) {
        bridge->config = *config;
    } else {
        stp_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep_system;
    bridge->effects.release_probability_factor = 1.0f;
    bridge->effects.depression_recovery_factor = 1.0f;
    bridge->effects.facilitation_decay_factor = 1.0f;
    bridge->effects.vesicle_restoration_active = false;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep_system,
        stp_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for STP bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep_system);
    stp_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("STP-sleep bridge created");
    return bridge;
}

void stp_sleep_bridge_destroy(stp_sleep_bridge_t bridge) {
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            stp_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for STP bridge");
        }
    }

    if (bridge->mutex) nimcp_platform_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int stp_sleep_update(stp_sleep_bridge_t bridge) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    if (bridge->config.enable_u_modulation) {
        float u_base = stp_sleep_get_u_factor(state);
        bridge->effects.release_probability_factor =
            1.0f + (u_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_tau_d_modulation) {
        float tau_d_base = stp_sleep_get_tau_d_factor(state);
        bridge->effects.depression_recovery_factor =
            1.0f + (tau_d_base - 1.0f) * bridge->config.modulation_strength;
    }

    if (bridge->config.enable_tau_f_modulation) {
        float tau_f_base = stp_sleep_get_tau_f_factor(state);
        bridge->effects.facilitation_decay_factor =
            1.0f + (tau_f_base - 1.0f) * bridge->config.modulation_strength;
    }

    bridge->effects.vesicle_restoration_active = (state == SLEEP_STATE_DEEP_NREM);

    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int stp_sleep_get_effects(const stp_sleep_bridge_t bridge,
                           stp_sleep_effects_t* effects) {
    if (!bridge || !effects) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

float stp_sleep_get_release_probability(const stp_sleep_bridge_t bridge,
                                         float base_u) {
    if (!bridge) return base_u;
    nimcp_platform_mutex_lock(bridge->mutex);
    float result = base_u * bridge->effects.release_probability_factor;
    /* Clamp to valid range [0, 1] */
    if (result > 1.0f) result = 1.0f;
    if (result < 0.0f) result = 0.0f;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return result;
}

float stp_sleep_get_tau_depression(const stp_sleep_bridge_t bridge,
                                    float base_tau_d) {
    if (!bridge) return base_tau_d;
    nimcp_platform_mutex_lock(bridge->mutex);
    /* Higher factor = faster recovery (shorter effective tau) */
    float result = base_tau_d / bridge->effects.depression_recovery_factor;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return result;
}

float stp_sleep_get_tau_facilitation(const stp_sleep_bridge_t bridge,
                                      float base_tau_f) {
    if (!bridge) return base_tau_f;
    nimcp_platform_mutex_lock(bridge->mutex);
    /* Higher factor = slower decay (longer tau) */
    float result = base_tau_f * bridge->effects.facilitation_decay_factor;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return result;
}

float stp_sleep_get_u_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return STP_SLEEP_U_AWAKE;
        case SLEEP_STATE_DROWSY:     return STP_SLEEP_U_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return STP_SLEEP_U_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return STP_SLEEP_U_DEEP_NREM;
        case SLEEP_STATE_REM:        return STP_SLEEP_U_REM;
        default:                     return STP_SLEEP_U_AWAKE;
    }
}

float stp_sleep_get_tau_d_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return STP_SLEEP_TAU_D_AWAKE;
        case SLEEP_STATE_DROWSY:     return STP_SLEEP_TAU_D_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return STP_SLEEP_TAU_D_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return STP_SLEEP_TAU_D_DEEP_NREM;
        case SLEEP_STATE_REM:        return STP_SLEEP_TAU_D_REM;
        default:                     return STP_SLEEP_TAU_D_AWAKE;
    }
}

float stp_sleep_get_tau_f_factor(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:      return STP_SLEEP_TAU_F_AWAKE;
        case SLEEP_STATE_DROWSY:     return STP_SLEEP_TAU_F_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return STP_SLEEP_TAU_F_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return STP_SLEEP_TAU_F_DEEP_NREM;
        case SLEEP_STATE_REM:        return STP_SLEEP_TAU_F_REM;
        default:                     return STP_SLEEP_TAU_F_AWAKE;
    }
}
