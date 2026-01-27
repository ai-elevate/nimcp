/**
 * @file nimcp_surprise_substrate_bridge.c
 * @brief Bridge between Surprise Amplifier and metabolic substrate
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Metabolic constraints on surprise processing
 * WHY:  Low ATP reduces sensitivity; fatigue increases thresholds
 * HOW:  ATP/fatigue levels → surprise parameter modulation
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_substrate_bridge.h"
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

static nimcp_health_agent_t* g_surprise_substrate_health_agent = NULL;

void surprise_substrate_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_substrate_health_agent = agent;
}

static inline void surprise_substrate_heartbeat(const char* op, float progress) {
    if (g_surprise_substrate_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_substrate_health_agent, op, progress);
    }
}

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct surprise_substrate_bridge {
    surprise_substrate_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;
    void* substrate_system;

    /* State */
    surprise_substrate_effects_t effects;
    surprise_substrate_stats_t stats;

    /* Bio-async */
    void* router;
    bool bio_async_connected;

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Health agent */
    nimcp_health_agent_t* health_agent;

    /* Last known substrate state */
    float last_atp;
    float last_fatigue;

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

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

surprise_substrate_config_t surprise_substrate_bridge_default_config(void) {
    surprise_substrate_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.detection_sensitivity_mult = SURPRISE_SUBSTRATE_DEFAULT_DETECT_MULT;
    cfg.amplification_accuracy_mult = SURPRISE_SUBSTRATE_DEFAULT_AMPLIFY_MULT;
    cfg.decay_modulation_mult = SURPRISE_SUBSTRATE_DEFAULT_DECAY_MULT;
    cfg.refractory_modulation_mult = SURPRISE_SUBSTRATE_DEFAULT_REFRACT_MULT;
    cfg.min_capacity = SURPRISE_SUBSTRATE_DEFAULT_MIN_CAPACITY;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_substrate_bridge_t* surprise_substrate_bridge_create(
    const surprise_substrate_config_t* config)
{
    surprise_substrate_bridge_t* bridge = (surprise_substrate_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_substrate_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_SUBSTRATE_ERROR_NO_MEMORY,
                           sizeof(surprise_substrate_bridge_t),
                           "surprise_substrate_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_substrate_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_substrate_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_SUBSTRATE_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_substrate_bridge mutex allocation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize effects to full capacity */
    bridge->effects.detection_sensitivity = 1.0f;
    bridge->effects.amplification_accuracy = 1.0f;
    bridge->effects.decay_modulation = 1.0f;
    bridge->effects.refractory_modulation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    bridge->last_atp = 1.0f;
    bridge->last_fatigue = 0.0f;
    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-substrate bridge (min_cap=%.2f)",
                           bridge->config.min_capacity);
    }

    return bridge;
}

void surprise_substrate_bridge_destroy(surprise_substrate_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-substrate bridge (updates=%lu, warnings=%lu)",
                           (unsigned long)bridge->stats.modulation_updates,
                           (unsigned long)bridge->stats.capacity_warnings);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_substrate_bridge_connect_amplifier(
    surprise_substrate_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-substrate bridge connected to amplifier");
    }
    return 0;
}

int surprise_substrate_bridge_connect_substrate(
    surprise_substrate_bridge_t* bridge,
    void* substrate)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in connect_substrate");
    NIMCP_CHECK_THROW_IMMUNE(substrate != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL substrate in connect_substrate");

    nimcp_mutex_lock(bridge->mutex);
    bridge->substrate_system = substrate;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-substrate bridge connected to substrate system");
    }
    return 0;
}

int surprise_substrate_bridge_register_bio_async(
    surprise_substrate_bridge_t* bridge,
    void* router)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in register_bio_async");

    nimcp_mutex_lock(bridge->mutex);
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Operations API
 * ============================================================================ */

int surprise_substrate_bridge_update(
    surprise_substrate_bridge_t* bridge,
    float atp_level,
    float fatigue_level)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in update");

    surprise_substrate_heartbeat("update", 0.0f);

    float atp = clamp_f(atp_level, 0.0f, 1.0f);
    float fatigue = clamp_f(fatigue_level, 0.0f, 1.0f);

    nimcp_mutex_lock(bridge->mutex);

    bridge->last_atp = atp;
    bridge->last_fatigue = fatigue;

    /* Compute modulation effects from substrate state */
    /* High ATP → better detection; low ATP → worse */
    bridge->effects.detection_sensitivity = bridge->config.detection_sensitivity_mult * atp;
    bridge->effects.detection_sensitivity = clamp_f(
        bridge->effects.detection_sensitivity, bridge->config.min_capacity, 2.0f);

    /* High ATP → accurate amplification */
    bridge->effects.amplification_accuracy = bridge->config.amplification_accuracy_mult * atp;
    bridge->effects.amplification_accuracy = clamp_f(
        bridge->effects.amplification_accuracy, bridge->config.min_capacity, 2.0f);

    /* High fatigue → faster decay */
    bridge->effects.decay_modulation = bridge->config.decay_modulation_mult * (1.0f + fatigue);

    /* High fatigue → longer refractory */
    bridge->effects.refractory_modulation = bridge->config.refractory_modulation_mult * (1.0f + fatigue);

    /* Overall capacity = ATP * (1 - fatigue) */
    bridge->effects.overall_capacity = clamp_f(
        atp * (1.0f - fatigue * 0.5f), bridge->config.min_capacity, 1.0f);

    bridge->stats.modulation_updates++;

    if (bridge->effects.overall_capacity < 0.5f) {
        bridge->stats.capacity_warnings++;
    }
    if (atp < 0.2f) {
        bridge->stats.atp_critical_events++;
    }

    bridge->stats.total_updates++;
    bridge->update_count++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_substrate_bridge_get_effects(
    const surprise_substrate_bridge_t* bridge,
    surprise_substrate_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in get_effects");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL effects_out in get_effects");

    *effects_out = bridge->effects;
    return 0;
}

int surprise_substrate_bridge_apply_effects(
    surprise_substrate_bridge_t* bridge)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in apply_effects");

    surprise_substrate_heartbeat("apply_effects", 0.0f);

    /* Effects would be applied to the amplifier here via its API */
    /* For now, effects are available via get_effects for the amplifier to query */

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_substrate_bridge_get_stats(
    const surprise_substrate_bridge_t* bridge,
    surprise_substrate_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

int surprise_substrate_bridge_set_health_agent(
    surprise_substrate_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SUBSTRATE_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
