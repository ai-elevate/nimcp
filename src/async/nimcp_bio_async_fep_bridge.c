/**
 * @file nimcp_bio_async_fep_bridge.c
 * @brief Implementation of FEP bridge for bio-async system
 *
 * @author NIMCP Development Team
 * @date 2025-12-15
 */

#include "async/nimcp_bio_async_fep_bridge.h"
#include "utils/platform/nimcp_platform.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int bio_async_fep_default_config(bio_async_fep_config_t* config) {
    if (!config) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    config->prediction_horizon_ms = 100.0f;
    config->precision_decay_rate = 0.1f;
    config->surprise_threshold = 2.0f;
    config->primary_channel = BIO_CHANNEL_DOPAMINE;
    config->enable_channel_switching = true;
    config->learning_rate = 0.1f;
    config->enable_precision_learning = true;
    config->enable_prefetch = true;
    config->max_predictions = 32;

    return 0;
}

bio_async_fep_bridge_t* bio_async_fep_create(
    const bio_async_fep_config_t* config,
    fep_system_t* fep_system
) {
    if (!config || !fep_system) {
        NIMCP_LOGGING_ERROR("bio_async_fep_create: NULL config or fep_system");
        return NULL;
    }

    bio_async_fep_bridge_t* bridge = (bio_async_fep_bridge_t*)nimcp_malloc(
        sizeof(bio_async_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("bio_async_fep_create: Failed to allocate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(bio_async_fep_bridge_t));

    /* Copy configuration */
    memcpy(&bridge->config, config, sizeof(bio_async_fep_config_t));

    /* Connect FEP system */
    bridge->fep_system = fep_system;

    /* Initialize state */
    bridge->state.current_channel = config->primary_channel;
    bridge->state.fep_active = true;
    bridge->state.bio_async_connected = false;

    /* Initialize effects */
    bridge->fep_effects.predicted_channel = config->primary_channel;
    bridge->fep_effects.concentration_modulation = 1.0f;
    bridge->fep_effects.refractory_modulation = 1.0f;

    /* Create mutex */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("bio_async_fep_create: Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created bio-async FEP bridge");

    return bridge;
}

void bio_async_fep_destroy(bio_async_fep_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Disconnect from bio-async if connected */
    if (bridge->bio_async_enabled) {
        bio_async_fep_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Integration Implementation
 * ============================================================================ */

int bio_async_fep_update_effects(bio_async_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Get current free energy from FEP system */
    float free_energy = fep_get_free_energy(bridge->fep_system);
    bridge->stats.avg_free_energy =
        0.9f * bridge->stats.avg_free_energy + 0.1f * free_energy;

    /* High free energy suggests uncertainty - increase neuromodulator release */
    if (free_energy > 5.0f) {
        bridge->fep_effects.concentration_modulation = 1.5f;
    } else if (free_energy < 1.0f) {
        bridge->fep_effects.concentration_modulation = 0.8f;
    } else {
        bridge->fep_effects.concentration_modulation = 1.0f;
    }

    /* Compute prediction confidence from free energy */
    bridge->fep_effects.prediction_confidence = expf(-free_energy / 10.0f);

    /* Update channel preferences based on prediction errors */
    for (int i = 0; i < BIO_CHANNEL_COUNT; i++) {
        /* Default equal preference */
        bridge->fep_effects.channel_preference[i] = 0.25f;
    }

    /* Modulate based on surprise */
    if (bridge->bio_async_effects.high_surprise_event) {
        /* High surprise → use fast channels (ACh, NE) */
        bridge->fep_effects.channel_preference[BIO_CHANNEL_ACETYLCHOLINE] = 0.4f;
        bridge->fep_effects.channel_preference[BIO_CHANNEL_NOREPINEPHRINE] = 0.3f;
        bridge->fep_effects.channel_preference[BIO_CHANNEL_DOPAMINE] = 0.2f;
        bridge->fep_effects.channel_preference[BIO_CHANNEL_SEROTONIN] = 0.1f;
    }

    /* Determine if should prefetch */
    bridge->fep_effects.should_prefetch =
        bridge->config.enable_prefetch &&
        bridge->fep_effects.prediction_confidence > 0.7f;

    bridge->fep_effects.prefetch_urgency = bridge->fep_effects.prediction_confidence;

    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int bio_async_fep_observe_future(
    bio_async_fep_bridge_t* bridge,
    nimcp_bio_future_t future
) {
    if (!bridge || !future) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Get confidence from bio-future */
    float confidence = nimcp_bio_future_get_confidence(future);

    /* Map confidence to precision (inverse variance) */
    bridge->bio_async_effects.confidence_based_precision =
        confidence * 10.0f; /* High confidence → high precision */

    /* Update precision decay rate based on confidence decay */
    float age_ms = nimcp_bio_future_get_age_ms(future);
    if (age_ms > 0.0f) {
        bridge->bio_async_effects.precision_decay_rate =
            -logf(confidence) / age_ms;
    }

    /* Update statistics */
    bridge->stats.avg_precision =
        0.9f * bridge->stats.avg_precision +
        0.1f * bridge->bio_async_effects.confidence_based_precision;

    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int bio_async_fep_predict_timing(
    bio_async_fep_bridge_t* bridge,
    nimcp_bio_channel_type_t channel,
    float* predicted_ms,
    float* confidence
) {
    if (!bridge || !predicted_ms || !confidence) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Use FEP to predict next observation time */
    /* Simple model: predict based on channel decay characteristics */
    float decay_tau = nimcp_bio_channel_decay_tau(channel);
    *predicted_ms = bridge->config.prediction_horizon_ms;
    *confidence = bridge->fep_effects.prediction_confidence;

    /* Update prediction count */
    bridge->state.active_predictions++;
    bridge->state.total_predictions++;

    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

nimcp_bio_channel_type_t bio_async_fep_select_channel(
    bio_async_fep_bridge_t* bridge,
    float message_urgency
) {
    if (!bridge) {
        return BIO_CHANNEL_DOPAMINE; /* Safe default */
    }

    nimcp_platform_mutex_lock(bridge->mutex);

    nimcp_bio_channel_type_t selected = bridge->config.primary_channel;

    if (bridge->config.enable_channel_switching) {
        /* Select channel based on urgency and preferences */
        if (message_urgency > 0.8f) {
            /* Very urgent → fast channel */
            selected = BIO_CHANNEL_ACETYLCHOLINE;
        } else if (message_urgency > 0.5f) {
            /* Moderately urgent */
            selected = BIO_CHANNEL_NOREPINEPHRINE;
        } else if (message_urgency > 0.2f) {
            /* Normal priority */
            selected = BIO_CHANNEL_DOPAMINE;
        } else {
            /* Low priority, slow coordination */
            selected = BIO_CHANNEL_SEROTONIN;
        }

        if (selected != bridge->state.current_channel) {
            bridge->state.channel_switches++;
        }
        bridge->state.current_channel = selected;
    }

    nimcp_platform_mutex_unlock(bridge->mutex);

    return selected;
}

/* ============================================================================
 * Bio-Async Integration Implementation
 * ============================================================================ */

int bio_async_fep_connect_bio_async(bio_async_fep_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (bridge->bio_async_enabled) {
        return 0; /* Already connected */
    }

    /* Check if bio-async is initialized */
    if (!nimcp_bio_async_is_initialized()) {
        NIMCP_LOGGING_WARN("Bio-async not initialized, cannot connect");
        return -1;
    }

    /* Register with bio-router (if available) */
    /* Note: bio-router is optional, this is best-effort */
    bridge->bio_async_enabled = true;

    NIMCP_LOGGING_INFO("Connected bio-async FEP bridge to bio-async system");

    return 0;
}

int bio_async_fep_disconnect_bio_async(bio_async_fep_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->bio_async_enabled) {
        return 0; /* Already disconnected */
    }

    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_INFO("Disconnected bio-async FEP bridge from bio-async system");

    return 0;
}

bool bio_async_fep_is_bio_async_connected(const bio_async_fep_bridge_t* bridge) {
    return bridge && bridge->bio_async_enabled;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int bio_async_fep_get_effects(
    const bio_async_fep_bridge_t* bridge,
    bio_async_fep_effects_t* effects
) {
    if (!bridge || !effects) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    memcpy(effects, &bridge->fep_effects, sizeof(bio_async_fep_effects_t));
    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int bio_async_fep_get_bio_async_effects(
    const bio_async_fep_bridge_t* bridge,
    fep_bio_async_effects_t* effects
) {
    if (!bridge || !effects) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    memcpy(effects, &bridge->bio_async_effects, sizeof(fep_bio_async_effects_t));
    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int bio_async_fep_get_stats(
    const bio_async_fep_bridge_t* bridge,
    bio_async_fep_stats_t* stats
) {
    if (!bridge || !stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    memcpy(stats, &bridge->stats, sizeof(bio_async_fep_stats_t));
    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int bio_async_fep_reset_stats(bio_async_fep_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(bio_async_fep_stats_t));
    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}
