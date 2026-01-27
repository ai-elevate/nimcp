/**
 * @file nimcp_surprise_snn_bridge.c
 * @brief Bridge between Surprise Amplifier and Spiking Neural Network
 * @version 1.0.0
 * @date 2026-01-27
 *
 * WHAT: Encode surprise as spike trains; decode SNN activity as surprise effects
 * WHY:  Spiking networks provide temporal precision for surprise encoding
 * HOW:  Surprise magnitude → firing rate; SNN patterns → channel activations
 *
 * @author NIMCP Development Team
 */

#include "cognitive/salience/nimcp_surprise_snn_bridge.h"
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

static nimcp_health_agent_t* g_surprise_snn_health_agent = NULL;

void surprise_snn_bridge_set_health_agent_global(nimcp_health_agent_t* agent) {
    g_surprise_snn_health_agent = agent;
}

static inline void surprise_snn_heartbeat(const char* op, float progress) {
    if (g_surprise_snn_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_surprise_snn_health_agent, op, progress);
    }
}

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

typedef struct {
    float membrane_potential;
    float refractory_remaining;
    bool spiked;
} snn_neuron_t;

struct surprise_snn_bridge {
    surprise_snn_config_t config;

    /* Connected systems */
    struct surprise_amplifier* amplifier;
    void* snn_system;

    /* State */
    surprise_snn_effects_t effects;
    surprise_snn_stats_t stats;

    /* Neuron arrays (one per channel) */
    snn_neuron_t* neurons;  /* neurons_per_channel * NUM_CHANNELS */
    uint32_t total_neurons;

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

static inline snn_neuron_t* get_channel_neurons(surprise_snn_bridge_t* bridge,
                                                  surprise_snn_channel_t ch) {
    return &bridge->neurons[(uint32_t)ch * bridge->config.neurons_per_channel];
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

surprise_snn_config_t surprise_snn_bridge_default_config(void) {
    surprise_snn_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.dt_ms = SURPRISE_SNN_DEFAULT_DT_MS;
    cfg.neurons_per_channel = SURPRISE_SNN_DEFAULT_NEURONS_PER_CH;
    cfg.encoding_type = SURPRISE_SNN_ENCODING_RATE;
    cfg.threshold = SURPRISE_SNN_DEFAULT_THRESHOLD;
    cfg.refractory_ms = SURPRISE_SNN_DEFAULT_REFRACTORY_MS;
    cfg.decay_factor = SURPRISE_SNN_DEFAULT_DECAY_FACTOR;
    cfg.history_size = SURPRISE_SNN_DEFAULT_HISTORY_SIZE;
    cfg.channel_weights[SURPRISE_SNN_CHANNEL_PE] = 1.0f;
    cfg.channel_weights[SURPRISE_SNN_CHANNEL_CONFLICT] = 1.0f;
    cfg.channel_weights[SURPRISE_SNN_CHANNEL_NOVELTY] = 1.0f;
    cfg.channel_weights[SURPRISE_SNN_CHANNEL_HYPOTHESIS] = 1.0f;
    cfg.enable_bio_async = true;
    cfg.enable_logging = true;

    return cfg;
}

surprise_snn_bridge_t* surprise_snn_bridge_create(
    const surprise_snn_config_t* config)
{
    surprise_snn_bridge_t* bridge = (surprise_snn_bridge_t*)nimcp_calloc(
        1, sizeof(surprise_snn_bridge_t));
    if (!bridge) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_SNN_ERROR_NO_MEMORY,
                           sizeof(surprise_snn_bridge_t),
                           "surprise_snn_bridge allocation failed (%zu bytes)",
                           sizeof(surprise_snn_bridge_t));
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = surprise_snn_bridge_default_config();
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_SNN_ERROR_NO_MEMORY,
                           sizeof(nimcp_mutex_t),
                           "surprise_snn_bridge mutex allocation failed");
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate neurons */
    bridge->total_neurons = bridge->config.neurons_per_channel * SURPRISE_SNN_NUM_CHANNELS;
    bridge->neurons = (snn_neuron_t*)nimcp_calloc(
        bridge->total_neurons, sizeof(snn_neuron_t));
    if (!bridge->neurons) {
        NIMCP_THROW_MEMORY(NIMCP_SURPRISE_SNN_ERROR_NO_MEMORY,
                           bridge->total_neurons * sizeof(snn_neuron_t),
                           "surprise_snn_bridge neurons allocation failed");
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->initialized = true;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Created surprise-SNN bridge (%u neurons, %u per channel)",
                           bridge->total_neurons, bridge->config.neurons_per_channel);
    }

    return bridge;
}

void surprise_snn_bridge_destroy(surprise_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Destroying surprise-SNN bridge (spikes=%lu, encodings=%lu)",
                           (unsigned long)bridge->stats.total_spikes,
                           (unsigned long)bridge->stats.encoding_events);
    }

    if (bridge->neurons) {
        nimcp_free(bridge->neurons);
    }
    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }
    nimcp_free(bridge);
}

int surprise_snn_bridge_reset(surprise_snn_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in reset");

    surprise_snn_heartbeat("reset", 0.0f);

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->effects, 0, sizeof(bridge->effects));
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    memset(bridge->neurons, 0, bridge->total_neurons * sizeof(snn_neuron_t));
    bridge->update_count = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int surprise_snn_bridge_connect_amplifier(
    surprise_snn_bridge_t* bridge,
    struct surprise_amplifier* amp)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in connect_amplifier");
    NIMCP_CHECK_THROW_IMMUNE(amp != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL amp in connect_amplifier");

    nimcp_mutex_lock(bridge->mutex);
    bridge->amplifier = amp;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-SNN bridge connected to amplifier");
    }
    return 0;
}

int surprise_snn_bridge_connect_snn(
    surprise_snn_bridge_t* bridge,
    void* snn)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in connect_snn");
    NIMCP_CHECK_THROW_IMMUNE(snn != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL snn in connect_snn");

    nimcp_mutex_lock(bridge->mutex);
    bridge->snn_system = snn;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge->config.enable_logging) {
        NIMCP_LOGGING_INFO("Surprise-SNN bridge connected to SNN system");
    }
    return 0;
}

int surprise_snn_bridge_connect_bio_async(
    surprise_snn_bridge_t* bridge,
    void* router)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in connect_bio_async");

    nimcp_mutex_lock(bridge->mutex);
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_snn_bridge_disconnect_bio_async(
    surprise_snn_bridge_t* bridge)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
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

int surprise_snn_encode_surprise(
    surprise_snn_bridge_t* bridge,
    float surprise_level,
    surprise_snn_channel_t channel)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in encode_surprise");
    NIMCP_CHECK_THROW_IMMUNE((uint32_t)channel < SURPRISE_SNN_NUM_CHANNELS,
                             NIMCP_SURPRISE_SNN_ERROR_INVALID_PARAM,
                             "Invalid channel %u in encode_surprise", (unsigned)channel);

    surprise_snn_heartbeat("encode_surprise", 0.0f);

    float level = clamp_f(surprise_level, 0.0f, 1.0f);

    nimcp_mutex_lock(bridge->mutex);

    /* Inject current into channel neurons based on surprise level */
    snn_neuron_t* ch_neurons = get_channel_neurons(bridge, channel);
    float weight = bridge->config.channel_weights[(uint32_t)channel];
    float injection = level * weight;

    for (uint32_t i = 0; i < bridge->config.neurons_per_channel; i++) {
        if (ch_neurons[i].refractory_remaining <= 0.0f) {
            ch_neurons[i].membrane_potential += injection;
        }
    }

    bridge->stats.encoding_events++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_snn_simulate_step(surprise_snn_bridge_t* bridge) {
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in simulate_step");

    surprise_snn_heartbeat("simulate_step", 0.0f);

    nimcp_mutex_lock(bridge->mutex);

    uint32_t spikes_this_step = 0;
    uint32_t channel_spikes[SURPRISE_SNN_NUM_CHANNELS] = {0};

    for (uint32_t ch = 0; ch < SURPRISE_SNN_NUM_CHANNELS; ch++) {
        snn_neuron_t* neurons = get_channel_neurons(bridge, (surprise_snn_channel_t)ch);

        for (uint32_t i = 0; i < bridge->config.neurons_per_channel; i++) {
            if ((i & 0xFF) == 0 && bridge->config.neurons_per_channel > 256) {
                surprise_snn_heartbeat("simulate_step", (float)i / bridge->config.neurons_per_channel);
            }

            snn_neuron_t* n = &neurons[i];
            n->spiked = false;

            if (n->refractory_remaining > 0.0f) {
                n->refractory_remaining -= bridge->config.dt_ms;
                continue;
            }

            /* Leak */
            n->membrane_potential *= bridge->config.decay_factor;

            /* Check threshold */
            if (n->membrane_potential >= bridge->config.threshold) {
                n->spiked = true;
                n->membrane_potential = 0.0f;
                n->refractory_remaining = bridge->config.refractory_ms;
                spikes_this_step++;
                channel_spikes[ch]++;
            }
        }
    }

    /* Update stats */
    bridge->stats.total_spikes += spikes_this_step;
    for (uint32_t ch = 0; ch < SURPRISE_SNN_NUM_CHANNELS; ch++) {
        bridge->stats.channel_spikes[ch] += channel_spikes[ch];
    }

    /* Update effects */
    float total_possible = (float)(bridge->config.neurons_per_channel * SURPRISE_SNN_NUM_CHANNELS);
    bridge->effects.spike_rate = (float)spikes_this_step / total_possible;
    bridge->effects.combined_activity = bridge->effects.spike_rate;
    bridge->effects.high_activity_flag = (bridge->effects.spike_rate > 0.5f);

    /* Compute per-channel activations and find dominant */
    float max_activation = 0.0f;
    bridge->effects.dominant_channel = SURPRISE_SNN_CHANNEL_PE;
    for (uint32_t ch = 0; ch < SURPRISE_SNN_NUM_CHANNELS; ch++) {
        float act = (float)channel_spikes[ch] / (float)bridge->config.neurons_per_channel;
        bridge->effects.channel_activations[ch] = act;
        if (act > max_activation) {
            max_activation = act;
            bridge->effects.dominant_channel = (surprise_snn_channel_t)ch;
        }
    }
    bridge->effects.confidence = (max_activation > 0.0f) ?
        max_activation / (bridge->effects.spike_rate + 0.001f) : 0.0f;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int surprise_snn_decode_output(
    surprise_snn_bridge_t* bridge,
    surprise_snn_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in decode_output");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL effects_out in decode_output");

    *effects_out = bridge->effects;
    return 0;
}

surprise_snn_channel_t surprise_snn_get_dominant_channel(
    const surprise_snn_bridge_t* bridge)
{
    if (!bridge) return SURPRISE_SNN_CHANNEL_PE;
    return bridge->effects.dominant_channel;
}

int surprise_snn_bridge_update(
    surprise_snn_bridge_t* bridge,
    float dt_seconds)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in update");

    if (dt_seconds <= 0.0f) return 0;

    surprise_snn_heartbeat("update", 0.0f);

    /* Run simulation steps for the elapsed time */
    float time_remaining = dt_seconds * 1000.0f; /* convert to ms */
    while (time_remaining > 0.0f) {
        surprise_snn_simulate_step(bridge);
        time_remaining -= bridge->config.dt_ms;
    }

    nimcp_mutex_lock(bridge->mutex);
    bridge->stats.total_updates++;
    bridge->update_count++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int surprise_snn_bridge_get_effects(
    const surprise_snn_bridge_t* bridge,
    surprise_snn_effects_t* effects_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in get_effects");
    NIMCP_CHECK_THROW_IMMUNE(effects_out != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL effects_out in get_effects");

    *effects_out = bridge->effects;
    return 0;
}

int surprise_snn_bridge_get_stats(
    const surprise_snn_bridge_t* bridge,
    surprise_snn_stats_t* stats_out)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in get_stats");
    NIMCP_CHECK_THROW_IMMUNE(stats_out != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL stats_out in get_stats");

    *stats_out = bridge->stats;
    return 0;
}

/* ============================================================================
 * Health Agent Integration
 * ============================================================================ */

int surprise_snn_bridge_set_health_agent(
    surprise_snn_bridge_t* bridge,
    struct nimcp_health_agent* agent)
{
    NIMCP_CHECK_THROW_IMMUNE(bridge != NULL,
                             NIMCP_SURPRISE_SNN_ERROR_NULL_POINTER,
                             "NULL bridge in set_health_agent");

    bridge->health_agent = agent;
    return 0;
}
