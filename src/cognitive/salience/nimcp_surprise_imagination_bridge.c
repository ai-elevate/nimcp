/**
 * @file nimcp_surprise_imagination_bridge.c
 * @brief Bridge between Surprise Amplifier and imagination/counterfactual system
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: High surprise triggers counterfactual imagination scenarios
 * WHY:  Surprising events should trigger "what if" reasoning
 * HOW:  Surprise threshold → trigger imagination; results → expectation update
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_imagination_bridge.h"
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

static nimcp_health_agent_t* g_surprise_imagination_health_agent = NULL;

void surprise_imagination_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_imagination_health_agent = agent;
}

static inline void surprise_imagination_heartbeat(const char* op, float progress) {
    if (g_surprise_imagination_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_imagination_health_agent, op, progress);
    }
}

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct surprise_imagination_bridge {
    surprise_imagination_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;
    void* imagination_engine;

    /* State */
    surprise_imagination_effects_t effects;
    surprise_imagination_stats_t stats;

    /* Scenario management */
    surprise_imagination_scenario_t* scenarios;
    uint32_t next_scenario_id;
    float cooldown_timer;

    /* Bio-async */
    void* router;
    bool bio_async_connected;

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

static surprise_imagination_scenario_t* find_scenario(
    surprise_imagination_bridge_t* bridge, uint32_t scenario_id)
{
    for (uint32_t i = 0; i < bridge->config.max_scenarios; i++) {
        if (bridge->scenarios[i].scenario_id == scenario_id &&
            bridge->scenarios[i].status != SURPRISE_IMAGINATION_STATUS_CANCELLED) {
            return &bridge->scenarios[i];
        }
    }
    return NULL;
}

static surprise_imagination_scenario_t* find_free_slot(
    surprise_imagination_bridge_t* bridge)
{
    for (uint32_t i = 0; i < bridge->config.max_scenarios; i++) {
        if (bridge->scenarios[i].status == SURPRISE_IMAGINATION_STATUS_COMPLETED ||
            bridge->scenarios[i].status == SURPRISE_IMAGINATION_STATUS_CANCELLED ||
            bridge->scenarios[i].scenario_id == 0) {
            return &bridge->scenarios[i];
        }
    }
    return NULL;
}

static uint32_t count_active(surprise_imagination_bridge_t* bridge) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < bridge->config.max_scenarios; i++) {
        if (bridge->scenarios[i].status == SURPRISE_IMAGINATION_STATUS_ACTIVE ||
            bridge->scenarios[i].status == SURPRISE_IMAGINATION_STATUS_PENDING) {
            count++;
        }
    }
    return count;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

surprise_imagination_config_t surprise_imagination_bridge_default_config(void) {
    surprise_imagination_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.trigger_threshold = SURPRISE_IMAGINATION_DEFAULT_TRIGGER_THRESHOLD;
    cfg.cooldown_seconds = SURPRISE_IMAGINATION_DEFAULT_COOLDOWN_SECONDS;
    cfg.max_scenarios = SURPRISE_IMAGINATION_DEFAULT_MAX_SCENARIOS;
    cfg.expectation_update_rate = SURPRISE_IMAGINATION_DEFAULT_EXPECT_UPDATE_RATE;
    cfg.counterfactual_depth = SURPRISE_IMAGINATION_DEFAULT_CF_DEPTH;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_imagination_bridge_t* surprise_imagination_bridge_create(
    const surprise_imagination_config_t* config)
{
    surprise_imagination_bridge_t* bridge = (surprise_imagination_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_imagination_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_IMAGINATION_ERROR_NO_MEMORY,
                           sizeof(surprise_imagination_bridge_t),
                           "surprise_imagination_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_imagination_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_imagination_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_IMAGINATION_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_imagination_bridge mutex allocation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate scenario slots */
    bridge->scenarios = (surprise_imagination_scenario_t*)nimcp_calloc(
        bridge->config.max_scenarios, sizeof(surprise_imagination_scenario_t));
    if (!bridge->scenarios) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_IMAGINATION_ERROR_NO_MEMORY,
                           bridge->config.max_scenarios * sizeof(surprise_imagination_scenario_t),
                           "surprise_imagination_bridge scenarios allocation failed");
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->next_scenario_id = 1;
    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-imagination bridge (threshold=%.2f, max_scenarios=%u)",
                           bridge->config.trigger_threshold, bridge->config.max_scenarios);
    }

    return bridge;
}

void surprise_imagination_bridge_destroy(surprise_imagination_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-imagination bridge (triggers=%lu, completed=%lu)",
                           (unsigned long)bridge->stats.triggers,
                           (unsigned long)bridge->stats.scenarios_completed);
    }

    if (bridge->scenarios) {
        nimcp_free(bridge->scenarios);
    }
    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

int surprise_imagination_bridge_reset(surprise_imagination_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in reset");

    surprise_imagination_heartbeat("reset", 0.0f);

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->effects, 0, sizeof(bridge->effects));
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    memset(bridge->scenarios, 0,
           bridge->config.max_scenarios * sizeof(surprise_imagination_scenario_t));
    bridge->next_scenario_id = 1;
    bridge->cooldown_timer = 0.0f;
    bridge->update_count = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_imagination_bridge_connect_amplifier(
    surprise_imagination_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-imagination bridge connected to amplifier");
    }
    return 0;
}

int surprise_imagination_bridge_connect_imagination_engine(
    surprise_imagination_bridge_t* bridge,
    void* engine)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in connect_imagination_engine");
    NIMCP_CHECK_THROW_IMMUNE(engine != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL engine in connect_imagination_engine");

    nimcp_mutex_lock(bridge->mutex);
    bridge->imagination_engine = engine;
    bridge->effects.imagination_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-imagination bridge connected to imagination engine");
    }
    return 0;
}

int surprise_imagination_bridge_connect_bio_async(
    surprise_imagination_bridge_t* bridge,
    void* router)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in connect_bio_async");

    nimcp_mutex_lock(bridge->mutex);
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_imagination_bridge_disconnect_bio_async(
    surprise_imagination_bridge_t* bridge)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
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

int surprise_imagination_check_trigger(
    surprise_imagination_bridge_t* bridge,
    float surprise_level,
    uint32_t source_module)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in check_trigger");

    surprise_imagination_heartbeat("check_trigger", 0.0f);

    float level = clamp_f(surprise_level, 0.0f, 1.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Check cooldown */
    if (bridge->cooldown_timer > 0.0f) {
        bridge->stats.cooldown_blocked++;
        bridge->effects.cooldown_remaining = bridge->cooldown_timer;
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Check threshold */
    if (level < bridge->config.trigger_threshold) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Check available slots */
    surprise_imagination_scenario_t* slot = find_free_slot(bridge);
    if (!slot) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Create new scenario */
    memset(slot, 0, sizeof(*slot));
    slot->scenario_id = bridge->next_scenario_id++;
    slot->trigger_magnitude = level;
    slot->trigger_source = source_module;
    slot->status = SURPRISE_IMAGINATION_STATUS_ACTIVE;
    slot->expected_outcome = level * 0.5f;  /* Initial expectation */

    /* Start cooldown */
    bridge->cooldown_timer = bridge->config.cooldown_seconds;

    /* Update effects */
    bridge->effects.scenarios_active = count_active(bridge);
    bridge->effects.last_trigger_magnitude = level;
    bridge->effects.cooldown_remaining = bridge->cooldown_timer;

    bridge->stats.triggers++;

    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Imagination triggered: scenario=%u, surprise=%.3f, source=0x%x",
                           slot->scenario_id, level, source_module);
    }

    return 0;
}

int surprise_imagination_on_result(
    surprise_imagination_bridge_t* bridge,
    uint32_t scenario_id,
    float actual_outcome)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in on_result");

    surprise_imagination_heartbeat("on_result", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    surprise_imagination_scenario_t* scenario = find_scenario(bridge, scenario_id);
    if (!scenario) {
        nimcp_mutex_unlock(bridge->mutex);
        return NIMCP_SURPRISE_IMAGINATION_ERROR_INVALID_PARAM;
    }

    scenario->actual_outcome = actual_outcome;
    scenario->divergence = fabsf(scenario->expected_outcome - actual_outcome);
    scenario->status = SURPRISE_IMAGINATION_STATUS_COMPLETED;

    /* Update expectation adjustment based on divergence */
    bridge->effects.expectation_adjustment +=
        bridge->config.expectation_update_rate * scenario->divergence;

    bridge->effects.scenarios_active = count_active(bridge);
    bridge->stats.scenarios_completed++;
    bridge->stats.expectations_updated++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_imagination_bridge_update(
    surprise_imagination_bridge_t* bridge,
    float dt_seconds)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in update");

    if (dt_seconds <= 0.0f) return 0;

    surprise_imagination_heartbeat("update", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Decay cooldown */
    if (bridge->cooldown_timer > 0.0f) {
        bridge->cooldown_timer -= dt_seconds;
        if (bridge->cooldown_timer < 0.0f) {
            bridge->cooldown_timer = 0.0f;
        }
    }
    bridge->effects.cooldown_remaining = bridge->cooldown_timer;

    /* Decay expectation adjustment toward zero */
    bridge->effects.expectation_adjustment *= 0.99f;

    bridge->effects.scenarios_active = count_active(bridge);
    bridge->stats.total_updates++;
    bridge->update_count++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_imagination_bridge_get_effects(
    const surprise_imagination_bridge_t* bridge,
    surprise_imagination_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in get_effects");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL effects_out in get_effects");

    *effects_out = bridge->effects;
    return 0;
}

int surprise_imagination_bridge_get_stats(
    const surprise_imagination_bridge_t* bridge,
    surprise_imagination_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

int surprise_imagination_get_scenario(
    const surprise_imagination_bridge_t* bridge,
    uint32_t scenario_id,
    surprise_imagination_scenario_t* scenario_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in get_scenario");
    NIMCP_CHECK_THROW_IMMUNE(scenario_out != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL scenario_out in get_scenario");

    for (uint32_t i = 0; i < bridge->config.max_scenarios; i++) {
        if (bridge->scenarios[i].scenario_id == scenario_id) {
            *scenario_out = bridge->scenarios[i];
            return 0;
        }
    }

    return NIMCP_SURPRISE_IMAGINATION_ERROR_INVALID_PARAM;
}

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

int surprise_imagination_bridge_set_health_agent(
    surprise_imagination_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_IMAGINATION_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
