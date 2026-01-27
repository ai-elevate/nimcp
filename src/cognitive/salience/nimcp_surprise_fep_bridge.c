/**
 * @file nimcp_surprise_fep_bridge.c
 * @brief Bridge between Surprise Amplifier and Free Energy Principle system
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Bidirectional integration: FEP PE → surprise amplifier; surprise → FEP precision
 * WHY:  FEP prediction errors are the primary driver of surprise signals;
 *       high surprise modulates FEP precision (attention = precision weighting)
 * HOW:  Forwards PE/KL to amplifier; reads surprise level to modulate precision
 *
 * NIMCP INTEGRATION PATTERNS:
 * 1. Exception handling (NIMCP_CHECK_THROW_IMMUNE)
 * 2. Health agent heartbeat
 * 3. Thread safety (mutex)
 * 4. Logging
 * 5. Memory management (nimcp_calloc/free)
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_fep_bridge.h"
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

static nimcp_health_agent_t* g_surprise_fep_bridge_health_agent = NULL;

void surprise_fep_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_fep_bridge_health_agent = agent;
}

static inline void surprise_fep_bridge_heartbeat(const char* op, float progress) {
    if (g_surprise_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_fep_bridge_health_agent, op, progress);
    }
}

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct surprise_fep_bridge {
    surprise_fep_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;
    void* fep_system;

    /* State */
    surprise_fep_effects_t effects;
    surprise_fep_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Health agent (per-instance) */
    nimcp_health_agent_t* health_agent;

    bool initialized;
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

surprise_fep_config_t surprise_fep_bridge_default_config(void) {
    surprise_fep_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.precision_gain = SURPRISE_FEP_DEFAULT_PRECISION_GAIN;
    cfg.pe_weight = SURPRISE_FEP_DEFAULT_PE_WEIGHT;
    cfg.bayesian_weight = SURPRISE_FEP_DEFAULT_BAYESIAN_WEIGHT;
    cfg.precision_floor = SURPRISE_FEP_DEFAULT_PRECISION_FLOOR;
    cfg.precision_ceiling = SURPRISE_FEP_DEFAULT_PRECISION_CEILING;
    cfg.pe_threshold = SURPRISE_FEP_DEFAULT_PE_THRESHOLD;
    cfg.enable_precision_modulation = true;
    cfg.enable_pe_forwarding = true;
    cfg.enable_bayesian_forwarding = true;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_fep_bridge_t* surprise_fep_bridge_create(
    const surprise_fep_config_t* config)
{
    surprise_fep_bridge_t* bridge = (surprise_fep_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_fep_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_FEP_ERROR_NO_MEMORY,
                           sizeof(surprise_fep_bridge_t),
                           "surprise_fep_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_fep_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_fep_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_FEP_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_fep_bridge mutex allocation failed (%zu bytes)",
                           sizeof(nimcp_mutex_t));
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.current_precision_boost = 1.0f;
    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-FEP bridge (pe_thresh=%.2f, prec_gain=%.1f)",
                           bridge->config.pe_threshold, bridge->config.precision_gain);
    }

    return bridge;
}

void surprise_fep_bridge_destroy(surprise_fep_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-FEP bridge (forwarded=%lu PE, %lu Bayesian)",
                           (unsigned long)bridge->stats.pe_events_forwarded,
                           (unsigned long)bridge->stats.bayesian_events_forwarded);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

int surprise_fep_bridge_reset(surprise_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in reset");

    surprise_fep_bridge_heartbeat("reset", 0.0f);

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->effects, 0, sizeof(bridge->effects));
    bridge->effects.current_precision_boost = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_fep_bridge_connect_amplifier(
    surprise_fep_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-FEP bridge connected to amplifier");
    }
    return 0;
}

int surprise_fep_bridge_connect_fep(
    surprise_fep_bridge_t* bridge,
    void* fep)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in connect_fep");
    NIMCP_CHECK_THROW_IMMUNE(fep != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL fep in connect_fep");

    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-FEP bridge connected to FEP system");
    }
    return 0;
}

/* ============================================================================
 * Operations API
 * ============================================================================ */

int surprise_fep_forward_pe(
    surprise_fep_bridge_t* bridge,
    float prediction_error,
    uint32_t source_module)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in forward_pe");

    surprise_fep_bridge_heartbeat("forward_pe", 0.0f);

    if (!bridge->config.enable_pe_forwarding) {
        return 0;
    }

    float pe = clamp_f(prediction_error, 0.0f, 1.0f);

    /* Threshold check */
    if (pe < bridge->config.pe_threshold) {
        nimcp_mutex_lock(bridge->mutex);
        bridge->stats.pe_below_threshold++;
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Forward to amplifier if connected */
    nimcp_mutex_lock(bridge->mutex);
    if (bridge->amplifier) {
        float weighted = pe * bridge->config.pe_weight;
        int rc = surprise_amplifier_on_prediction_error(
            bridge->amplifier, weighted, source_module);
        if (rc == 0) {
            bridge->stats.pe_events_forwarded++;
            bridge->effects.last_pe_forwarded = weighted;

            /* Running average */
            float n = (float)bridge->stats.pe_events_forwarded;
            bridge->stats.avg_pe_forwarded =
                bridge->stats.avg_pe_forwarded * ((n - 1.0f) / n) + weighted / n;
        }
    }
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_fep_forward_bayesian(
    surprise_fep_bridge_t* bridge,
    float kl_divergence,
    uint32_t source_module)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in forward_bayesian");

    surprise_fep_bridge_heartbeat("forward_bayesian", 0.0f);

    if (!bridge->config.enable_bayesian_forwarding) {
        return 0;
    }

    if (kl_divergence < 0.0f) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_SURPRISE_FEP_ERROR_INVALID_PARAM,
                              "Negative KL divergence: %f", kl_divergence);
        return NIMCP_SURPRISE_FEP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_lock(bridge->mutex);
    if (bridge->amplifier) {
        float weighted = kl_divergence * bridge->config.bayesian_weight;
        int rc = surprise_amplifier_on_bayesian_surprise(
            bridge->amplifier, weighted, source_module);
        if (rc == 0) {
            bridge->stats.bayesian_events_forwarded++;
            bridge->effects.last_bayesian_forwarded = weighted;
        }
    }
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_fep_modulate_precision(surprise_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in modulate_precision");

    surprise_fep_bridge_heartbeat("modulate_precision", 0.0f);

    if (!bridge->config.enable_precision_modulation) {
        return 0;
    }

    nimcp_mutex_lock(bridge->mutex);

    /* Read current surprise level from amplifier */
    float surprise_level = 0.0f;
    if (bridge->amplifier) {
        surprise_level = surprise_amplifier_get_current_level(bridge->amplifier);
    }

    /* Compute precision boost: base + surprise * gain */
    float boost = 1.0f + surprise_level * bridge->config.precision_gain;
    boost = clamp_f(boost, bridge->config.precision_floor,
                    bridge->config.precision_ceiling);

    bridge->effects.current_precision_boost = boost;
    bridge->effects.integrated_surprise = surprise_level;
    bridge->stats.precision_modulations++;

    /* Running avg */
    float n = (float)bridge->stats.precision_modulations;
    bridge->stats.avg_precision_boost =
        bridge->stats.avg_precision_boost * ((n - 1.0f) / n) + boost / n;
    if (boost > bridge->stats.max_precision_boost) {
        bridge->stats.max_precision_boost = boost;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float surprise_fep_get_precision_boost(const surprise_fep_bridge_t* bridge) {
    if (!bridge) return 1.0f;
    return bridge->effects.current_precision_boost;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int surprise_fep_bridge_update(surprise_fep_bridge_t* bridge, float dt_seconds) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in update");

    if (dt_seconds <= 0.0f) return 0;

    surprise_fep_bridge_heartbeat("update", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Auto-modulate precision on update */
    if (bridge->config.enable_precision_modulation && bridge->amplifier) {
        float surprise_level = surprise_amplifier_get_current_level(bridge->amplifier);
        float boost = 1.0f + surprise_level * bridge->config.precision_gain;
        boost = clamp_f(boost, bridge->config.precision_floor,
                        bridge->config.precision_ceiling);
        bridge->effects.current_precision_boost = boost;
        bridge->effects.integrated_surprise = surprise_level;
    }

    bridge->stats.total_updates++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_fep_bridge_get_effects(
    const surprise_fep_bridge_t* bridge,
    surprise_fep_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in get_effects");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL effects_out in get_effects");

    *effects_out = bridge->effects;
    return 0;
}

int surprise_fep_bridge_get_stats(
    const surprise_fep_bridge_t* bridge,
    surprise_fep_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

/* ============================================================================
 * Health Agent Integration (per-instance)
 * ============================================================================ */

int surprise_fep_bridge_set_health_agent(
    surprise_fep_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_FEP_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
