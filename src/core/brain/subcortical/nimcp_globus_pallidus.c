//=============================================================================
// nimcp_globus_pallidus.c - Globus Pallidus Implementation
//=============================================================================
/**
 * @file nimcp_globus_pallidus.c
 * @brief Globus pallidus implementation (GPe and GPi)
 *
 * WHAT: GP model with tonic inhibition and disinhibition dynamics
 * WHY:  GP is the output stage controlling thalamic activity
 * HOW:  High tonic firing provides constant inhibition; striatal input modulates
 *
 * @author NIMCP Development Team
 * @date 2025-12-22
 */

#include "core/brain/subcortical/nimcp_globus_pallidus.h"
#include "api/nimcp_api_exception.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>
#include <math.h>

//=============================================================================
// Lifecycle Functions
//=============================================================================

void globus_pallidus_default_config(globus_pallidus_config_t* config,
                                     gp_segment_t segment) {
    if (!config) return;

    config->segment = segment;
    config->num_neurons = GP_DEFAULT_NEURONS;
    config->num_actions = 8;
    config->tonic_firing_rate = GP_TONIC_FIRING_RATE;
    config->max_firing_rate = GP_MAX_FIRING_RATE;
    config->inhibition_time_const = GP_INHIBITION_TIME_CONST;
    config->inhibition_gain = 1.0f;
}

globus_pallidus_t* globus_pallidus_create(const globus_pallidus_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config for globus pallidus");
        return NULL;
    }

    globus_pallidus_t* gp = nimcp_malloc(sizeof(globus_pallidus_t));
    if (!gp) {
        NIMCP_LOGGING_ERROR("Failed to allocate globus pallidus");
        return NULL;
    }
    memset(gp, 0, sizeof(globus_pallidus_t));

    gp->segment = config->segment;
    gp->num_neurons = config->num_neurons;
    gp->num_actions = config->num_actions;
    gp->config = *config;

    /* Allocate neurons */
    gp->neurons = nimcp_calloc(config->num_neurons, sizeof(gp_neuron_t));
    if (!gp->neurons) {
        NIMCP_LOGGING_ERROR("Failed to allocate GP neurons");
        nimcp_free(gp);
        return NULL;
    }

    /* Initialize neurons with tonic firing */
    uint32_t neurons_per_action = config->num_neurons / config->num_actions;
    for (uint32_t i = 0; i < config->num_neurons; i++) {
        gp->neurons[i].id = i;
        gp->neurons[i].firing_rate = config->tonic_firing_rate;
        gp->neurons[i].membrane_potential = -60.0f;
        gp->neurons[i].inhibition = 0.0f;
        gp->neurons[i].target_action = i / neurons_per_action;
        if (gp->neurons[i].target_action >= config->num_actions) {
            gp->neurons[i].target_action = config->num_actions - 1;
        }
    }

    /* Allocate output buffer */
    gp->output = nimcp_calloc(config->num_actions, sizeof(float));
    if (!gp->output) {
        NIMCP_LOGGING_ERROR("Failed to allocate GP output buffer");
        nimcp_free(gp->neurons);
        nimcp_free(gp);
        return NULL;
    }

    /* Initialize output with tonic firing (normalized) */
    for (uint32_t a = 0; a < config->num_actions; a++) {
        gp->output[a] = config->tonic_firing_rate / config->max_firing_rate;
    }

    /* Allocate input buffers */
    gp->striatal_input = nimcp_calloc(config->num_actions, sizeof(float));
    gp->stn_input = nimcp_calloc(config->num_actions, sizeof(float));
    gp->gpe_input = nimcp_calloc(config->num_actions, sizeof(float));

    if (!gp->striatal_input || !gp->stn_input || !gp->gpe_input) {
        NIMCP_LOGGING_ERROR("Failed to allocate GP input buffers");
        if (gp->striatal_input) nimcp_free(gp->striatal_input);
        if (gp->stn_input) nimcp_free(gp->stn_input);
        if (gp->gpe_input) nimcp_free(gp->gpe_input);
        nimcp_free(gp->output);
        nimcp_free(gp->neurons);
        nimcp_free(gp);
        return NULL;
    }

    /* Allocate mutex */
    gp->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!gp->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate GP mutex");
        nimcp_free(gp->striatal_input);
        nimcp_free(gp->stn_input);
        nimcp_free(gp->gpe_input);
        nimcp_free(gp->output);
        nimcp_free(gp->neurons);
        nimcp_free(gp);
        return NULL;
    }
    nimcp_mutex_init(gp->mutex, NULL);

    NIMCP_LOGGING_DEBUG("Created globus pallidus %s with %u neurons, %u actions",
                        globus_pallidus_segment_name(config->segment),
                        config->num_neurons, config->num_actions);

    return gp;
}

void globus_pallidus_destroy(globus_pallidus_t* gp) {
    if (!gp) return;

    nimcp_mutex_lock(gp->mutex);

    nimcp_free(gp->neurons);
    nimcp_free(gp->output);
    nimcp_free(gp->striatal_input);
    nimcp_free(gp->stn_input);
    nimcp_free(gp->gpe_input);

    nimcp_mutex_unlock(gp->mutex);
    nimcp_mutex_destroy(gp->mutex);
    nimcp_free(gp->mutex);

    nimcp_free(gp);

    NIMCP_LOGGING_DEBUG("Destroyed globus pallidus");
}

int globus_pallidus_reset(globus_pallidus_t* gp) {
    if (!gp) return -1;

    nimcp_mutex_lock(gp->mutex);

    /* Reset neurons to tonic firing */
    for (uint32_t i = 0; i < gp->num_neurons; i++) {
        gp->neurons[i].firing_rate = gp->config.tonic_firing_rate;
        gp->neurons[i].inhibition = 0.0f;
    }

    /* Reset output */
    for (uint32_t a = 0; a < gp->num_actions; a++) {
        gp->output[a] = gp->config.tonic_firing_rate / gp->config.max_firing_rate;
    }

    /* Clear inputs */
    memset(gp->striatal_input, 0, gp->num_actions * sizeof(float));
    memset(gp->stn_input, 0, gp->num_actions * sizeof(float));
    memset(gp->gpe_input, 0, gp->num_actions * sizeof(float));

    nimcp_mutex_unlock(gp->mutex);

    return 0;
}

//=============================================================================
// Processing Functions
//=============================================================================

int globus_pallidus_set_striatal_input(globus_pallidus_t* gp,
                                        const float* inhibition) {
    if (!gp || !inhibition) return -1;

    nimcp_mutex_lock(gp->mutex);
    memcpy(gp->striatal_input, inhibition, gp->num_actions * sizeof(float));
    nimcp_mutex_unlock(gp->mutex);

    return 0;
}

int globus_pallidus_set_stn_input(globus_pallidus_t* gp,
                                   const float* excitation) {
    if (!gp || !excitation) return -1;

    nimcp_mutex_lock(gp->mutex);
    memcpy(gp->stn_input, excitation, gp->num_actions * sizeof(float));
    nimcp_mutex_unlock(gp->mutex);

    return 0;
}

int globus_pallidus_set_gpe_input(globus_pallidus_t* gp,
                                   const float* input) {
    if (!gp || !input) return -1;
    if (gp->segment != GP_SEGMENT_INTERNAL) {
        NIMCP_LOGGING_WARN("GPe input only valid for GPi");
        return -2;
    }

    nimcp_mutex_lock(gp->mutex);
    memcpy(gp->gpe_input, input, gp->num_actions * sizeof(float));
    nimcp_mutex_unlock(gp->mutex);

    return 0;
}

int globus_pallidus_process(globus_pallidus_t* gp) {
    if (!gp) return -1;

    nimcp_mutex_lock(gp->mutex);

    float tonic = gp->config.tonic_firing_rate;
    float max_rate = gp->config.max_firing_rate;
    float inhib_gain = gp->config.inhibition_gain;

    for (uint32_t a = 0; a < gp->num_actions; a++) {
        float striatal_inhib = gp->striatal_input[a] * inhib_gain;
        float stn_excite = gp->stn_input[a];

        float firing_rate;

        if (gp->segment == GP_SEGMENT_EXTERNAL) {
            /* GPe: inhibited by D2 striatal neurons */
            firing_rate = tonic * (1.0f - striatal_inhib);
        } else {
            /* GPi: inhibited by D1 striatal, excited by STN */
            /* Also receives GPe input (indirect pathway adds to inhibition) */
            float gpe_effect = gp->gpe_input[a];

            /* D1 inhibition reduces firing (direct pathway) */
            /* STN excitation increases firing (hyperdirect + indirect) */
            /* GPe reduction leads to STN disinhibition (indirect pathway) */
            firing_rate = tonic * (1.0f - striatal_inhib)
                        + (stn_excite * tonic * 0.5f)
                        - (gpe_effect * 0.2f);
        }

        /* Clamp firing rate */
        firing_rate = fmaxf(0.0f, fminf(max_rate, firing_rate));

        /* Normalize output to [0, 1] */
        gp->output[a] = firing_rate / max_rate;
    }

    /* Update statistics */
    float sum = 0, min_rate = max_rate, max_rate_actual = 0;
    for (uint32_t a = 0; a < gp->num_actions; a++) {
        float rate = gp->output[a] * max_rate;
        sum += rate;
        if (rate < min_rate) min_rate = rate;
        if (rate > max_rate_actual) max_rate_actual = rate;
    }
    gp->stats.avg_firing_rate = sum / gp->num_actions;
    gp->stats.min_firing_rate = min_rate;
    gp->stats.max_firing_rate = max_rate_actual;

    nimcp_mutex_unlock(gp->mutex);

    return 0;
}

int globus_pallidus_get_output(const globus_pallidus_t* gp, float* output) {
    if (!gp || !output) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)gp->mutex);
    memcpy(output, gp->output, gp->num_actions * sizeof(float));
    nimcp_mutex_unlock((nimcp_mutex_t*)gp->mutex);

    return 0;
}

float globus_pallidus_get_action_output(const globus_pallidus_t* gp,
                                         uint32_t action_id) {
    if (!gp || action_id >= gp->num_actions) return -1.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)gp->mutex);
    float output = gp->output[action_id];
    nimcp_mutex_unlock((nimcp_mutex_t*)gp->mutex);

    return output;
}

int globus_pallidus_step(globus_pallidus_t* gp, float dt) {
    if (!gp) return -1;

    /* Process with current inputs */
    return globus_pallidus_process(gp);
}

int globus_pallidus_get_stats(const globus_pallidus_t* gp, gp_stats_t* stats) {
    if (!gp || !stats) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)gp->mutex);
    *stats = gp->stats;
    nimcp_mutex_unlock((nimcp_mutex_t*)gp->mutex);

    return 0;
}

const char* globus_pallidus_segment_name(gp_segment_t segment) {
    switch (segment) {
        case GP_SEGMENT_EXTERNAL: return "GPe";
        case GP_SEGMENT_INTERNAL: return "GPi";
        default: return "Unknown";
    }
}
