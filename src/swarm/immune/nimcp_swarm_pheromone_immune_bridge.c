/**
 * @file nimcp_swarm_pheromone_immune_bridge.c
 * @brief Swarm Pheromone-Immune System Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-18
 *
 * WHAT: Modulates swarm pheromone sensing based on immune state
 * WHY:  Inflammation affects pheromone sensing; contamination triggers immune
 * HOW:  Cytokines reduce sensing threshold; contamination triggers antigen response
 *
 * @author NIMCP Development Team
 */

#include "swarm/immune/nimcp_swarm_pheromone_immune_bridge.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <stdlib.h>
#include <string.h>

struct swarm_pheromone_immune_bridge_struct {
    swarm_pheromone_immune_config_t config;
    brain_immune_system_t* immune_system;
    void* swarm_pheromone;
    cytokine_pheromone_effects_t cytokine_effects;
    inflammation_pheromone_state_t inflammation_state;
    pheromone_immune_modulation_t modulation;
    nimcp_mutex_t* mutex;
    void* bio_ctx;
    bool bio_async_connected;
};

static float get_pheromone_factor_for_level(brain_inflammation_level_t level)
{
    switch (level) {
        case INFLAMMATION_NONE:     return INFLAMMATION_NONE_PHEROMONE_FACTOR;
        case INFLAMMATION_LOCAL:    return INFLAMMATION_LOCAL_PHEROMONE_FACTOR;
        case INFLAMMATION_REGIONAL: return INFLAMMATION_REGIONAL_PHEROMONE_FACTOR;
        case INFLAMMATION_SYSTEMIC: return INFLAMMATION_SYSTEMIC_PHEROMONE_FACTOR;
        case INFLAMMATION_STORM:    return INFLAMMATION_STORM_PHEROMONE_FACTOR;
        default:                          return INFLAMMATION_NONE_PHEROMONE_FACTOR;
    }
}

int swarm_pheromone_immune_default_config(swarm_pheromone_immune_config_t* config)
{
    if (!config) return -1;

    config->enable_cytokine_effects = true;
    config->enable_inflammation_effects = true;
    config->enable_contamination_detection = true;
    config->cytokine_sensitivity = 1.0f;

    return 0;
}

swarm_pheromone_immune_bridge_t* swarm_pheromone_immune_bridge_create(
    const swarm_pheromone_immune_config_t* config,
    brain_immune_system_t* immune_system,
    void* swarm_pheromone)
{
    if (!config || !immune_system) {
        NIMCP_LOGGING_ERROR("Invalid parameters for swarm pheromone immune bridge creation");
        return NULL;
    }

    swarm_pheromone_immune_bridge_t* bridge =
        (swarm_pheromone_immune_bridge_t*)nimcp_malloc(sizeof(swarm_pheromone_immune_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate swarm pheromone immune bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(swarm_pheromone_immune_bridge_t));
    memcpy(&bridge->config, config, sizeof(swarm_pheromone_immune_config_t));
    bridge->immune_system = immune_system;
    bridge->swarm_pheromone = swarm_pheromone;

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for swarm pheromone immune bridge");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->cytokine_effects.sensing_threshold_increase = 0.0f;
    bridge->cytokine_effects.evaporation_rate_increase = 0.0f;
    bridge->cytokine_effects.gradient_detection_reduction = 0.0f;

    bridge->inflammation_state.current_level = INFLAMMATION_NONE;
    bridge->inflammation_state.pheromone_factor = 1.0f;
    bridge->inflammation_state.sensing_efficiency = 1.0f;
    bridge->inflammation_state.gradient_impaired = false;

    bridge->modulation.contamination_events = 0;
    bridge->modulation.contamination_level = 0.0f;
    bridge->modulation.immune_activated = false;
    bridge->modulation.cleanup_signal = 0.0f;

    NIMCP_LOGGING_INFO("Swarm pheromone immune bridge created");
    return bridge;
}

void swarm_pheromone_immune_bridge_destroy(swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Swarm pheromone immune bridge destroyed");
}

int swarm_pheromone_immune_apply_cytokine_effects(swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge || !bridge->immune_system) return -1;
    if (!bridge->config.enable_cytokine_effects) return 0;

    nimcp_mutex_lock(bridge->mutex);

    float sensitivity = bridge->config.cytokine_sensitivity;

    bridge->cytokine_effects.sensing_threshold_increase =
        -(CYTOKINE_IL1_SENSING_REDUCTION +
          CYTOKINE_IL6_SENSING_REDUCTION +
          CYTOKINE_TNF_SENSING_REDUCTION +
          CYTOKINE_IL10_SENSING_BOOST) * sensitivity;

    bridge->cytokine_effects.evaporation_rate_increase =
        bridge->cytokine_effects.sensing_threshold_increase * 0.5f;
    bridge->cytokine_effects.gradient_detection_reduction =
        bridge->cytokine_effects.sensing_threshold_increase * 0.7f;

    if (bridge->cytokine_effects.sensing_threshold_increase < 0.0f) {
        bridge->cytokine_effects.sensing_threshold_increase = 0.0f;
    }

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Applied cytokine pheromone effects: threshold_increase=%.2f",
                        bridge->cytokine_effects.sensing_threshold_increase);
    return 0;
}

int swarm_pheromone_immune_apply_inflammation_effects(swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge || !bridge->immune_system) return -1;
    if (!bridge->config.enable_inflammation_effects) return 0;

    nimcp_mutex_lock(bridge->mutex);

    brain_immune_stats_t stats;
    if (brain_immune_get_stats(bridge->immune_system, &stats) != 0) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }
    brain_inflammation_level_t level = stats.inflammation_level;
    bridge->inflammation_state.current_level = level;
    bridge->inflammation_state.pheromone_factor = get_pheromone_factor_for_level(level);
    bridge->inflammation_state.sensing_efficiency =
        bridge->inflammation_state.pheromone_factor;
    bridge->inflammation_state.gradient_impaired =
        (level >= INFLAMMATION_REGIONAL);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Applied inflammation pheromone effects: level=%d, factor=%.2f",
                        level, bridge->inflammation_state.pheromone_factor);
    return 0;
}

int swarm_pheromone_immune_report_contamination(swarm_pheromone_immune_bridge_t* bridge, float level)
{
    if (!bridge) return -1;
    if (!bridge->config.enable_contamination_detection) return 0;

    nimcp_mutex_lock(bridge->mutex);

    bridge->modulation.contamination_events++;
    bridge->modulation.contamination_level = level;

    if (level > 0.5f) {
        bridge->modulation.immune_activated = true;
        NIMCP_LOGGING_INFO("Pheromone contamination activated immune: level=%.2f",
                          level);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_pheromone_immune_boost_from_clean_path(swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->modulation.cleanup_signal += 0.05f;
    bridge->modulation.contamination_level = 0.0f;
    bridge->modulation.immune_activated = false;

    NIMCP_LOGGING_DEBUG("Clean pheromone path boosted cleanup: total=%.2f",
                       bridge->modulation.cleanup_signal);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int swarm_pheromone_immune_bridge_update(swarm_pheromone_immune_bridge_t* bridge, uint64_t delta_ms)
{
    if (!bridge) return -1;
    (void)delta_ms;

    swarm_pheromone_immune_apply_cytokine_effects(bridge);
    swarm_pheromone_immune_apply_inflammation_effects(bridge);

    return 0;
}

float swarm_pheromone_immune_get_sensing_factor(const swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float factor = bridge->inflammation_state.sensing_efficiency;
    nimcp_mutex_unlock(bridge->mutex);

    return factor;
}

float swarm_pheromone_immune_get_evaporation_factor(const swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float factor = 1.0f + bridge->cytokine_effects.evaporation_rate_increase;
    nimcp_mutex_unlock(bridge->mutex);

    return factor;
}

bool swarm_pheromone_immune_is_gradient_impaired(const swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool impaired = bridge->inflammation_state.gradient_impaired;
    nimcp_mutex_unlock(bridge->mutex);

    return impaired;
}

int swarm_pheromone_immune_connect_bio_async(swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot connect to bio-async: NULL bridge");
        return -1;
    }

    if (bridge->bio_async_connected) {
        NIMCP_LOGGING_INFO("Swarm pheromone-immune bridge already connected to bio-async");
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_IMMUNE_SWARM_PHEROMONE,
        .module_name = "swarm_pheromone_immune_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_connected = true;
        NIMCP_LOGGING_INFO("Swarm pheromone-immune bridge connected to bio-async router");
    } else {
        NIMCP_LOGGING_INFO("Bio-async router not available, skipping registration");
    }

    return 0;
}

int swarm_pheromone_immune_disconnect_bio_async(swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot disconnect from bio-async: NULL bridge");
        return -1;
    }

    if (!bridge->bio_async_connected) {
        return 0;
    }

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_connected = false;
    NIMCP_LOGGING_INFO("Swarm pheromone-immune bridge disconnected from bio-async router");

    return 0;
}

bool swarm_pheromone_immune_is_bio_async_connected(const swarm_pheromone_immune_bridge_t* bridge)
{
    if (!bridge) return false;
    return bridge->bio_async_connected;
}
