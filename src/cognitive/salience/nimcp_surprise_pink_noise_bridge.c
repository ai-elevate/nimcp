/**
 * @file nimcp_surprise_pink_noise_bridge.c
 * @brief Bridge between Surprise Amplifier and Pink Noise system
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Biologically realistic 1/f noise for surprise baseline parameters
 * WHY:  Neural systems operate with 1/f noise; natural threshold fluctuations
 * HOW:  Pink noise → parameter injection; surprise level → amplitude adaptation
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_pink_noise_bridge.h"
#include "cognitive/salience/nimcp_surprise_amplifier.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

static nimcp_health_agent_t* g_surprise_pink_noise_health_agent = NULL;

void surprise_pink_noise_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_pink_noise_health_agent = agent;
}

static inline void surprise_pink_noise_heartbeat(const char* op, float progress) {
    if (g_surprise_pink_noise_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_pink_noise_health_agent, op, progress);
    }
}

/* ============================================================================
 * Pink Noise Generator (Voss-McCartney algorithm)
 * ============================================================================ */

#define PINK_NOISE_NUM_OCTAVES 8

typedef struct {
    float octave_values[PINK_NOISE_NUM_OCTAVES];
    uint32_t counter;
    float running_sum;
} pink_noise_gen_t;

static void pink_noise_gen_init(pink_noise_gen_t* gen) {
    memset(gen, 0, sizeof(*gen));
}

static float pink_noise_gen_next(pink_noise_gen_t* gen) {
    gen->counter++;

    /* Voss-McCartney: update octave based on trailing zeros */
    uint32_t changed = gen->counter;
    int octave = 0;
    while (octave < PINK_NOISE_NUM_OCTAVES && !(changed & 1)) {
        changed >>= 1;
        octave++;
    }

    if (octave < PINK_NOISE_NUM_OCTAVES) {
        /* Remove old value, generate new white noise, add it */
        gen->running_sum -= gen->octave_values[octave];
        float white = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        gen->octave_values[octave] = white;
        gen->running_sum += white;
    }

    /* Normalize by number of octaves */
    return gen->running_sum / (float)PINK_NOISE_NUM_OCTAVES;
}

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct surprise_pink_noise_bridge {
    surprise_pink_noise_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;

    /* State */
    surprise_pink_noise_effects_t effects;
    surprise_pink_noise_stats_t stats;

    /* Noise generators (one per target) */
    pink_noise_gen_t generators[SURPRISE_PINK_NOISE_NUM_TARGETS];

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

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

surprise_pink_noise_config_t surprise_pink_noise_bridge_default_config(void) {
    surprise_pink_noise_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.base_amplitude = SURPRISE_PINK_NOISE_DEFAULT_BASE_AMPLITUDE;
    cfg.alpha = SURPRISE_PINK_NOISE_DEFAULT_ALPHA;
    cfg.adaptation_rate = SURPRISE_PINK_NOISE_DEFAULT_ADAPT_RATE;
    cfg.target_amplitudes[SURPRISE_PINK_NOISE_TARGET_THRESHOLD] = 1.0f;
    cfg.target_amplitudes[SURPRISE_PINK_NOISE_TARGET_SENSITIVITY] = 0.8f;
    cfg.target_amplitudes[SURPRISE_PINK_NOISE_TARGET_DECAY] = 0.5f;
    cfg.target_amplitudes[SURPRISE_PINK_NOISE_TARGET_REFRACTORY] = 0.3f;
    cfg.temporal_smoothing = SURPRISE_PINK_NOISE_DEFAULT_SMOOTHING;
    cfg.min_amplitude = SURPRISE_PINK_NOISE_DEFAULT_MIN_AMPLITUDE;
    cfg.max_amplitude = SURPRISE_PINK_NOISE_DEFAULT_MAX_AMPLITUDE;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_pink_noise_bridge_t* surprise_pink_noise_bridge_create(
    const surprise_pink_noise_config_t* config)
{
    surprise_pink_noise_bridge_t* bridge = (surprise_pink_noise_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_pink_noise_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_PINK_NOISE_ERROR_NO_MEMORY,
                           sizeof(surprise_pink_noise_bridge_t),
                           "surprise_pink_noise_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_pink_noise_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_pink_noise_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_PINK_NOISE_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_pink_noise_bridge mutex allocation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize noise generators */
    for (int i = 0; i < SURPRISE_PINK_NOISE_NUM_TARGETS; i++) {
        pink_noise_gen_init(&bridge->generators[i]);
    }

    bridge->effects.effective_amplitude = bridge->config.base_amplitude;
    bridge->effects.adaptation_factor = 1.0f;
    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-pink noise bridge (amp=%.3f, alpha=%.1f)",
                           bridge->config.base_amplitude, bridge->config.alpha);
    }

    return bridge;
}

void surprise_pink_noise_bridge_destroy(surprise_pink_noise_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-pink noise bridge (injections=%lu, adaptations=%lu)",
                           (unsigned long)bridge->stats.noise_injections,
                           (unsigned long)bridge->stats.adaptations);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

int surprise_pink_noise_bridge_reset(surprise_pink_noise_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in reset");

    surprise_pink_noise_heartbeat("reset", 0.0f);

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->effects, 0, sizeof(bridge->effects));
    bridge->effects.effective_amplitude = bridge->config.base_amplitude;
    bridge->effects.adaptation_factor = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    for (int i = 0; i < SURPRISE_PINK_NOISE_NUM_TARGETS; i++) {
        pink_noise_gen_init(&bridge->generators[i]);
    }
    bridge->update_count = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_pink_noise_bridge_connect_amplifier(
    surprise_pink_noise_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-pink noise bridge connected to amplifier");
    }
    return 0;
}

int surprise_pink_noise_bridge_connect_bio_async(
    surprise_pink_noise_bridge_t* bridge,
    void* router)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in connect_bio_async");

    nimcp_mutex_lock(bridge->mutex);
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_pink_noise_bridge_disconnect_bio_async(
    surprise_pink_noise_bridge_t* bridge)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
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

int surprise_pink_noise_inject(surprise_pink_noise_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in inject");

    surprise_pink_noise_heartbeat("inject", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    float amplitude = bridge->effects.effective_amplitude;

    for (int i = 0; i < SURPRISE_PINK_NOISE_NUM_TARGETS; i++) {
        float raw = pink_noise_gen_next(&bridge->generators[i]);
        float target_scale = bridge->config.target_amplitudes[i];
        float noise = raw * amplitude * target_scale;

        /* Temporal smoothing (EMA) */
        bridge->effects.current_noise[i] =
            bridge->config.temporal_smoothing * bridge->effects.current_noise[i] +
            (1.0f - bridge->config.temporal_smoothing) * noise;
    }

    bridge->effects.samples_generated++;
    bridge->stats.noise_injections++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_pink_noise_adapt_amplitude(
    surprise_pink_noise_bridge_t* bridge,
    float surprise_level)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in adapt_amplitude");

    surprise_pink_noise_heartbeat("adapt_amplitude", 0.0f);

    float level = clamp_f(surprise_level, 0.0f, 1.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* High surprise → increase noise (more uncertainty) */
    float target_factor = 1.0f + level;
    bridge->effects.adaptation_factor +=
        bridge->config.adaptation_rate * (target_factor - bridge->effects.adaptation_factor);

    float old_amp = bridge->effects.effective_amplitude;
    bridge->effects.effective_amplitude = clamp_f(
        bridge->config.base_amplitude * bridge->effects.adaptation_factor,
        bridge->config.min_amplitude,
        bridge->config.max_amplitude);

    bridge->stats.adaptations++;

    if (fabsf(bridge->effects.effective_amplitude - old_amp) > 0.001f) {
        bridge->stats.amplitude_changes++;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_pink_noise_bridge_update(
    surprise_pink_noise_bridge_t* bridge,
    float dt_seconds)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in update");

    if (dt_seconds <= 0.0f) return 0;

    surprise_pink_noise_heartbeat("update", 0.0f);

    /* Inject noise on each update */
    surprise_pink_noise_inject(bridge);

    /* If connected to amplifier, adapt based on current surprise */
    nimcp_mutex_lock(bridge->mutex);
    if (bridge->amplifier) {
        float surprise = surprise_amplifier_get_current_level(bridge->amplifier);
        nimcp_mutex_unlock(bridge->mutex);
        surprise_pink_noise_adapt_amplitude(bridge, surprise);
        nimcp_mutex_lock(bridge->mutex);
    }

    bridge->stats.total_updates++;
    bridge->update_count++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float surprise_pink_noise_get_for_target(
    const surprise_pink_noise_bridge_t* bridge,
    uint32_t target)
{
    if (!bridge) return 0.0f;
    if (target >= SURPRISE_PINK_NOISE_NUM_TARGETS) return 0.0f;

    return bridge->effects.current_noise[target];
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_pink_noise_bridge_get_effects(
    const surprise_pink_noise_bridge_t* bridge,
    surprise_pink_noise_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in get_effects");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL effects_out in get_effects");

    *effects_out = bridge->effects;
    return 0;
}

int surprise_pink_noise_bridge_get_stats(
    const surprise_pink_noise_bridge_t* bridge,
    surprise_pink_noise_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

int surprise_pink_noise_bridge_set_health_agent(
    surprise_pink_noise_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_PINK_NOISE_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
