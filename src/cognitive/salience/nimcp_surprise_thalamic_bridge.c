/**
 * @file nimcp_surprise_thalamic_bridge.c
 * @brief Bridge between Surprise Amplifier and thalamic routing system
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Thalamic gating and routing of surprise signals
 * WHY:  Surprise signals must be routed to appropriate cortical destinations
 * HOW:  Signal type/urgency → routing destination; attention → gating weights
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_thalamic_bridge.h"
#include "cognitive/salience/nimcp_surprise_amplifier.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include <stddef.h>

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

static nimcp_health_agent_t* g_surprise_thalamic_health_agent = NULL;

void surprise_thalamic_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_thalamic_health_agent = agent;
}

static inline void surprise_thalamic_heartbeat(const char* op, float progress) {
    if (g_surprise_thalamic_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_thalamic_health_agent, op, progress);
    }
}

/* ============================================================================
 * Constants
 * ============================================================================ */

#define NUM_SIGNAL_TYPES 4

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct surprise_thalamic_bridge {
    surprise_thalamic_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;
    void* thalamic_router;

    /* State */
    surprise_thalamic_stats_t stats;

    /* Attention weights per signal type */
    float attention_weights[NUM_SIGNAL_TYPES];

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Health agent */
    nimcp_health_agent_t* health_agent;

    bool initialized;
    uint64_t update_count;
};

/* ============================================================================
 * Helpers
 * ============================================================================ */

static inline float clamp_f(float val, float lo, float hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

static int signal_type_to_index(uint32_t signal_type) {
    switch (signal_type) {
        case SURPRISE_THALAMIC_REALIZATION: return 0;
        case SURPRISE_THALAMIC_CONFLICT:    return 1;
        case SURPRISE_THALAMIC_NOVELTY:     return 2;
        case SURPRISE_THALAMIC_HYPOTHESIS:  return 3;
        default: return -1;
    }
}

static float get_threshold_for_type(const surprise_thalamic_config_t* cfg, uint32_t signal_type) {
    switch (signal_type) {
        case SURPRISE_THALAMIC_REALIZATION: return cfg->threshold_realization;
        case SURPRISE_THALAMIC_CONFLICT:    return cfg->threshold_conflict;
        case SURPRISE_THALAMIC_NOVELTY:     return cfg->threshold_novelty;
        case SURPRISE_THALAMIC_HYPOTHESIS:  return cfg->threshold_hypothesis;
        default: return 0.5f;
    }
}

static bool is_type_enabled(const surprise_thalamic_config_t* cfg, uint32_t signal_type) {
    switch (signal_type) {
        case SURPRISE_THALAMIC_REALIZATION: return cfg->enable_realization;
        case SURPRISE_THALAMIC_CONFLICT:    return cfg->enable_conflict;
        case SURPRISE_THALAMIC_NOVELTY:     return cfg->enable_novelty;
        case SURPRISE_THALAMIC_HYPOTHESIS:  return cfg->enable_hypothesis;
        default: return false;
    }
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

surprise_thalamic_config_t surprise_thalamic_bridge_default_config(void) {
    surprise_thalamic_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.enable_realization = true;
    cfg.enable_conflict = true;
    cfg.enable_novelty = true;
    cfg.enable_hypothesis = true;
    cfg.threshold_realization = 0.5f;
    cfg.threshold_conflict = 0.4f;
    cfg.threshold_novelty = 0.3f;
    cfg.threshold_hypothesis = 0.6f;
    cfg.attention_weight_default = SURPRISE_THALAMIC_DEFAULT_ATTENTION_WEIGHT;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_thalamic_bridge_t* surprise_thalamic_bridge_create(
    const surprise_thalamic_config_t* config)
{
    surprise_thalamic_bridge_t* bridge = (surprise_thalamic_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_thalamic_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_THALAMIC_ERROR_NO_MEMORY,
                           sizeof(surprise_thalamic_bridge_t),
                           "surprise_thalamic_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_thalamic_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_thalamic_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_THALAMIC_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_thalamic_bridge mutex allocation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize attention weights */
    for (int i = 0; i < NUM_SIGNAL_TYPES; i++) {
        bridge->attention_weights[i] = bridge->config.attention_weight_default;
    }

    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-thalamic bridge (attn_default=%.1f)",
                           bridge->config.attention_weight_default);
    }

    return bridge;
}

void surprise_thalamic_bridge_destroy(surprise_thalamic_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-thalamic bridge (routed=%lu, overrides=%lu)",
                           (unsigned long)bridge->stats.signals_routed,
                           (unsigned long)bridge->stats.overrides);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

int surprise_thalamic_bridge_reset(surprise_thalamic_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in reset");

    surprise_thalamic_heartbeat("reset", 0.0f);

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    for (int i = 0; i < NUM_SIGNAL_TYPES; i++) {
        bridge->attention_weights[i] = bridge->config.attention_weight_default;
    }
    bridge->update_count = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_thalamic_bridge_connect_amplifier(
    surprise_thalamic_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-thalamic bridge connected to amplifier");
    }
    return 0;
}

int surprise_thalamic_bridge_connect_thalamic_router(
    surprise_thalamic_bridge_t* bridge,
    void* router)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in connect_thalamic_router");
    NIMCP_CHECK_THROW_IMMUNE(router != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL router in connect_thalamic_router");

    nimcp_mutex_lock(bridge->mutex);
    bridge->thalamic_router = router;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-thalamic bridge connected to thalamic router");
    }
    return 0;
}

/* ============================================================================
 * Operations API
 * ============================================================================ */

int surprise_thalamic_route_surprise(
    surprise_thalamic_bridge_t* bridge,
    const surprise_thalamic_signal_t* signal)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in route_surprise");
    NIMCP_CHECK_THROW_IMMUNE(signal != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL signal in route_surprise");

    surprise_thalamic_heartbeat("route_surprise", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    uint32_t type = signal->signal_type;
    float magnitude = clamp_f(signal->surprise_magnitude, 0.0f, 1.0f);

    /* Check each signal type bit */
    uint32_t types[] = {SURPRISE_THALAMIC_REALIZATION, SURPRISE_THALAMIC_CONFLICT,
                        SURPRISE_THALAMIC_NOVELTY, SURPRISE_THALAMIC_HYPOTHESIS};

    for (int i = 0; i < NUM_SIGNAL_TYPES; i++) {
        if (!(type & types[i])) continue;
        if (!is_type_enabled(&bridge->config, types[i])) continue;

        int idx = signal_type_to_index(types[i]);
        if (idx < 0) continue;

        float threshold = get_threshold_for_type(&bridge->config, types[i]);
        float weighted_magnitude = magnitude * bridge->attention_weights[idx];

        if (weighted_magnitude >= threshold) {
            bridge->stats.signals_routed++;

            if (signal->urgency > 0.8f) {
                bridge->stats.high_priority_routes++;
            }
        }
    }

    /* Update running average */
    float n = (float)(bridge->stats.signals_routed > 0 ? bridge->stats.signals_routed : 1);
    bridge->stats.avg_surprise =
        bridge->stats.avg_surprise * ((n - 1.0f) / n) + magnitude / n;

    bridge->stats.total_updates++;
    bridge->update_count++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_thalamic_route_realization(
    surprise_thalamic_bridge_t* bridge,
    float magnitude,
    uint32_t source_module)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in route_realization");

    surprise_thalamic_signal_t signal;
    memset(&signal, 0, sizeof(signal));
    signal.signal_type = SURPRISE_THALAMIC_REALIZATION;
    signal.surprise_magnitude = magnitude;
    signal.source_module = source_module;
    signal.urgency = clamp_f(magnitude * 1.2f, 0.0f, 1.0f); /* Realizations are high priority */

    return surprise_thalamic_route_surprise(bridge, &signal);
}

int surprise_thalamic_set_attention_weight(
    surprise_thalamic_bridge_t* bridge,
    uint32_t signal_type,
    float weight)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in set_attention_weight");

    int idx = signal_type_to_index(signal_type);
    NIMCP_CHECK_THROW_IMMUNE(idx >= 0,
                             NIMCP_SURPRISE_THALAMIC_ERROR_INVALID_PARAM,
                             "Invalid signal type 0x%x", signal_type);

    nimcp_mutex_lock(bridge->mutex);
    bridge->attention_weights[idx] = clamp_f(weight, 0.0f, 5.0f);
    bridge->stats.gating_updates++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float surprise_thalamic_get_attention_weight(
    const surprise_thalamic_bridge_t* bridge,
    uint32_t signal_type)
{
    if (!bridge) return 1.0f;

    int idx = signal_type_to_index(signal_type);
    if (idx < 0) return 1.0f;

    return bridge->attention_weights[idx];
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_thalamic_bridge_get_stats(
    const surprise_thalamic_bridge_t* bridge,
    surprise_thalamic_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

int surprise_thalamic_bridge_set_health_agent(
    surprise_thalamic_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_THALAMIC_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
