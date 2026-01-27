/**
 * @file nimcp_surprise_plasticity_bridge.c
 * @brief Bridge between Surprise Amplifier and synaptic plasticity system
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Bidirectional: surprise → plasticity modulation; learning → surprise habituation
 * WHY:  High surprise boosts learning rate, STDP windows, eligibility traces;
 *       repeated learning reduces surprise sensitivity (habituation)
 * HOW:  Surprise events modulate plasticity params; learning outcomes habituate sources
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_plasticity_bridge.h"
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

static nimcp_health_agent_t* g_surprise_plasticity_health_agent = NULL;

void surprise_plasticity_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_plasticity_health_agent = agent;
}

static inline void surprise_plasticity_heartbeat(const char* op, float progress) {
    if (g_surprise_plasticity_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_plasticity_health_agent, op, progress);
    }
}

/* ============================================================================
 * Habituation Tracker
 * ============================================================================ */

typedef struct {
    uint32_t source_id;
    float habituation;  /* 0.0 = fully novel, 1.0 = fully habituated */
    uint64_t last_seen;
} habituation_entry_t;

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct surprise_plasticity_bridge {
    surprise_plasticity_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;
    void* plasticity_system;

    /* State */
    surprise_plasticity_effects_t effects;
    surprise_plasticity_stats_t stats;

    /* Habituation tracking */
    habituation_entry_t* sources;
    uint32_t source_count;

    /* Bio-async */
    void* router;
    bool bio_async_connected;

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Health agent (per-instance) */
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

static habituation_entry_t* find_source(surprise_plasticity_bridge_t* bridge, uint32_t source_id) {
    for (uint32_t i = 0; i < bridge->source_count; i++) {
        if (bridge->sources[i].source_id == source_id) {
            return &bridge->sources[i];
        }
    }
    return NULL;
}

static habituation_entry_t* add_source(surprise_plasticity_bridge_t* bridge, uint32_t source_id) {
    if (bridge->source_count >= bridge->config.max_tracked_sources) {
        /* Evict oldest (least recently seen) */
        uint32_t oldest_idx = 0;
        uint64_t oldest_time = bridge->sources[0].last_seen;
        for (uint32_t i = 1; i < bridge->source_count; i++) {
            if (bridge->sources[i].last_seen < oldest_time) {
                oldest_time = bridge->sources[i].last_seen;
                oldest_idx = i;
            }
        }
        bridge->sources[oldest_idx].source_id = source_id;
        bridge->sources[oldest_idx].habituation = 0.0f;
        bridge->sources[oldest_idx].last_seen = bridge->update_count;
        return &bridge->sources[oldest_idx];
    }
    habituation_entry_t* entry = &bridge->sources[bridge->source_count++];
    entry->source_id = source_id;
    entry->habituation = 0.0f;
    entry->last_seen = bridge->update_count;
    return entry;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

surprise_plasticity_config_t surprise_plasticity_bridge_default_config(void) {
    surprise_plasticity_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.learning_rate_boost = SURPRISE_PLASTICITY_DEFAULT_LR_BOOST;
    cfg.habituation_rate = SURPRISE_PLASTICITY_DEFAULT_HABITUATION_RATE;
    cfg.habituation_recovery_rate = SURPRISE_PLASTICITY_DEFAULT_HABITUATION_RECOVERY;
    cfg.stdp_window_expansion = SURPRISE_PLASTICITY_DEFAULT_STDP_EXPANSION;
    cfg.eligibility_boost = SURPRISE_PLASTICITY_DEFAULT_ELIGIBILITY_BOOST;
    cfg.bcm_threshold_shift = SURPRISE_PLASTICITY_DEFAULT_BCM_SHIFT;
    cfg.min_surprise_for_boost = SURPRISE_PLASTICITY_DEFAULT_MIN_SURPRISE;
    cfg.max_tracked_sources = SURPRISE_PLASTICITY_DEFAULT_MAX_SOURCES;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_plasticity_bridge_t* surprise_plasticity_bridge_create(
    const surprise_plasticity_config_t* config)
{
    surprise_plasticity_bridge_t* bridge = (surprise_plasticity_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_plasticity_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_PLASTICITY_ERROR_NO_MEMORY,
                           sizeof(surprise_plasticity_bridge_t),
                           "surprise_plasticity_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_plasticity_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_plasticity_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_PLASTICITY_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_plasticity_bridge mutex allocation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate habituation tracker */
    bridge->sources = (habituation_entry_t*)nimcp_calloc(
        bridge->config.max_tracked_sources, sizeof(habituation_entry_t));
    if (!bridge->sources) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_PLASTICITY_ERROR_NO_MEMORY,
                           bridge->config.max_tracked_sources * sizeof(habituation_entry_t),
                           "surprise_plasticity_bridge sources allocation failed");
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize effects to neutral */
    bridge->effects.learning_rate_multiplier = 1.0f;
    bridge->effects.stdp_window_multiplier = 1.0f;
    bridge->effects.eligibility_multiplier = 1.0f;
    bridge->effects.bcm_shift = 0.0f;
    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-plasticity bridge (lr_boost=%.1f, habit_rate=%.3f)",
                           bridge->config.learning_rate_boost,
                           bridge->config.habituation_rate);
    }

    return bridge;
}

void surprise_plasticity_bridge_destroy(surprise_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-plasticity bridge (boosts=%lu, habituation=%lu)",
                           (unsigned long)bridge->stats.plasticity_boosts,
                           (unsigned long)bridge->stats.habituation_events);
    }

    if (bridge->sources) {
        nimcp_free(bridge->sources);
    }
    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

int surprise_plasticity_bridge_reset(surprise_plasticity_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in reset");

    surprise_plasticity_heartbeat("reset", 0.0f);

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->effects, 0, sizeof(bridge->effects));
    bridge->effects.learning_rate_multiplier = 1.0f;
    bridge->effects.stdp_window_multiplier = 1.0f;
    bridge->effects.eligibility_multiplier = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    bridge->source_count = 0;
    memset(bridge->sources, 0,
           bridge->config.max_tracked_sources * sizeof(habituation_entry_t));
    bridge->update_count = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_plasticity_bridge_connect_amplifier(
    surprise_plasticity_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-plasticity bridge connected to amplifier");
    }
    return 0;
}

int surprise_plasticity_bridge_connect_plasticity(
    surprise_plasticity_bridge_t* bridge,
    void* plasticity)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in connect_plasticity");
    NIMCP_CHECK_THROW_IMMUNE(plasticity != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL plasticity in connect_plasticity");

    nimcp_mutex_lock(bridge->mutex);
    bridge->plasticity_system = plasticity;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-plasticity bridge connected to plasticity system");
    }
    return 0;
}

int surprise_plasticity_bridge_connect_bio_async(
    surprise_plasticity_bridge_t* bridge,
    void* router)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in connect_bio_async");

    nimcp_mutex_lock(bridge->mutex);
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_plasticity_bridge_disconnect_bio_async(
    surprise_plasticity_bridge_t* bridge)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in disconnect_bio_async");

    nimcp_mutex_lock(bridge->mutex);
    bridge->router = NULL;
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Operations API
 * ============================================================================ */

int surprise_plasticity_on_surprise_event(
    surprise_plasticity_bridge_t* bridge,
    float surprise_level,
    uint32_t source_id)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in on_surprise_event");

    surprise_plasticity_heartbeat("on_surprise_event", 0.0f);

    float level = clamp_f(surprise_level, 0.0f, 1.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Find or create habituation entry */
    habituation_entry_t* entry = find_source(bridge, source_id);
    if (!entry) {
        entry = add_source(bridge, source_id);
    }
    entry->last_seen = bridge->update_count;

    /* Apply habituation: effective surprise = raw * (1 - habituation) */
    float effective = level * (1.0f - entry->habituation);

    /* Update habituation (more exposure = more habituated) */
    entry->habituation = clamp_f(
        entry->habituation + bridge->config.habituation_rate, 0.0f, 1.0f);
    bridge->stats.habituation_events++;

    /* Check threshold */
    if (effective < bridge->config.min_surprise_for_boost) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Compute plasticity modulation */
    float surprise_factor = effective; /* 0 to 1 */
    bridge->effects.learning_rate_multiplier = 1.0f +
        surprise_factor * (bridge->config.learning_rate_boost - 1.0f);
    bridge->effects.stdp_window_multiplier = 1.0f +
        surprise_factor * (bridge->config.stdp_window_expansion - 1.0f);
    bridge->effects.eligibility_multiplier = 1.0f +
        surprise_factor * (bridge->config.eligibility_boost - 1.0f);
    bridge->effects.bcm_shift = surprise_factor * bridge->config.bcm_threshold_shift;
    bridge->effects.habituation_level = entry->habituation;
    bridge->effects.active_sources = bridge->source_count;

    bridge->stats.plasticity_boosts++;
    bridge->stats.learning_rate_updates++;

    if (bridge->effects.bcm_shift > 0.01f) {
        bridge->stats.bcm_shifts++;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_plasticity_on_learning_outcome(
    surprise_plasticity_bridge_t* bridge,
    float weight_change,
    uint32_t source_id)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in on_learning_outcome");

    surprise_plasticity_heartbeat("on_learning_outcome", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Learning outcome increases habituation (we learned about this) */
    habituation_entry_t* entry = find_source(bridge, source_id);
    if (entry) {
        float habit_increase = fabsf(weight_change) * bridge->config.habituation_rate;
        entry->habituation = clamp_f(entry->habituation + habit_increase, 0.0f, 1.0f);
        bridge->stats.habituation_events++;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_plasticity_bridge_update(
    surprise_plasticity_bridge_t* bridge,
    float dt_seconds)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in update");

    if (dt_seconds <= 0.0f) return 0;

    surprise_plasticity_heartbeat("update", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Decay habituation for sources not recently seen (recovery) */
    for (uint32_t i = 0; i < bridge->source_count; i++) {
        if ((bridge->update_count - bridge->sources[i].last_seen) > 10) {
            bridge->sources[i].habituation = clamp_f(
                bridge->sources[i].habituation - bridge->config.habituation_recovery_rate * dt_seconds,
                0.0f, 1.0f);
        }
    }

    /* Decay effects toward neutral */
    float decay = 0.95f;
    bridge->effects.learning_rate_multiplier =
        1.0f + (bridge->effects.learning_rate_multiplier - 1.0f) * decay;
    bridge->effects.stdp_window_multiplier =
        1.0f + (bridge->effects.stdp_window_multiplier - 1.0f) * decay;
    bridge->effects.eligibility_multiplier =
        1.0f + (bridge->effects.eligibility_multiplier - 1.0f) * decay;
    bridge->effects.bcm_shift *= decay;

    bridge->stats.total_updates++;
    bridge->update_count++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_plasticity_bridge_get_effects(
    const surprise_plasticity_bridge_t* bridge,
    surprise_plasticity_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in get_effects");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL effects_out in get_effects");

    *effects_out = bridge->effects;
    return 0;
}

int surprise_plasticity_bridge_get_stats(
    const surprise_plasticity_bridge_t* bridge,
    surprise_plasticity_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

float surprise_plasticity_get_habituation_for_source(
    const surprise_plasticity_bridge_t* bridge,
    uint32_t source_id)
{
    if (!bridge) return 0.0f;

    for (uint32_t i = 0; i < bridge->source_count; i++) {
        if (bridge->sources[i].source_id == source_id) {
            return bridge->sources[i].habituation;
        }
    }
    return 0.0f;  /* Unknown source = no habituation */
}

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

int surprise_plasticity_bridge_set_health_agent(
    surprise_plasticity_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PLASTICITY_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
