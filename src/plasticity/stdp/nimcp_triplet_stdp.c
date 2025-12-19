/**
 * @file nimcp_triplet_stdp.c
 * @brief Implementation of Triplet STDP (Pfister & Gerstner 2006)
 *
 * NIMCP Phase: Plasticity Module Enhancement
 * Date: 2025-12-19
 */

#include "plasticity/stdp/nimcp_triplet_stdp.h"
#include "plasticity/stdp/nimcp_triplet_stdp_sleep_bridge.h"
#include "plasticity/stdp/nimcp_triplet_stdp_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define LOG_MODULE "triplet_stdp"

/* ============================================================================
 * Configuration Presets
 * ============================================================================ */

triplet_stdp_config_t triplet_stdp_config_default(void) {
    /* WHAT: Return Pfister & Gerstner (2006) visual cortex parameters
     * WHY:  Provides validated starting point from experimental data
     * HOW:  Use published constants from visual cortex experiments
     */
    triplet_stdp_config_t config = {
        .A2_plus = TRIPLET_STDP_DEFAULT_A2_PLUS,
        .A3_plus = TRIPLET_STDP_DEFAULT_A3_PLUS,
        .A2_minus = TRIPLET_STDP_DEFAULT_A2_MINUS,
        .A3_minus = TRIPLET_STDP_DEFAULT_A3_MINUS,
        .tau_plus = TRIPLET_STDP_DEFAULT_TAU_PLUS,
        .tau_minus = TRIPLET_STDP_DEFAULT_TAU_MINUS,
        .tau_x = TRIPLET_STDP_DEFAULT_TAU_X,
        .tau_y = TRIPLET_STDP_DEFAULT_TAU_Y,
        .w_max = TRIPLET_STDP_DEFAULT_W_MAX,
        .w_min = TRIPLET_STDP_DEFAULT_W_MIN,
        .enable_bio_async = false,
        .enable_sleep_modulation = false,
        .enable_immune_modulation = false
    };
    return config;
}

triplet_stdp_config_t triplet_stdp_config_hippocampal(void) {
    /* WHAT: Return hippocampus-specific triplet parameters
     * WHY:  Hippocampus has different time constants and amplitudes
     * HOW:  Adjusted parameters for CA3-CA1 synapses
     */
    triplet_stdp_config_t config = triplet_stdp_config_default();

    /* Hippocampal synapses have faster dynamics */
    config.tau_plus = 15.0f;   /* Slightly faster */
    config.tau_minus = 30.0f;  /* Slightly faster */
    config.tau_x = 80.0f;      /* Faster slow trace */
    config.tau_y = 100.0f;     /* Faster slow trace */

    /* Stronger triplet terms in hippocampus */
    config.A3_plus = 0.01f;    /* More triplet LTP */
    config.A3_minus = 0.0005f; /* More triplet LTD */

    return config;
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

triplet_stdp_synapse_t* triplet_stdp_synapse_create(
    const triplet_stdp_config_t* config,
    float initial_weight
) {
    /* WHAT: Allocate and initialize triplet STDP synapse
     * WHY:  Proper resource allocation with NIMCP memory management
     * HOW:  Use nimcp_malloc, initialize fields, create mutex
     */

    /* Use default config if not provided */
    triplet_stdp_config_t default_config;
    if (!config) {
        default_config = triplet_stdp_config_default();
        config = &default_config;
    }

    /* Allocate synapse */
    triplet_stdp_synapse_t* synapse = (triplet_stdp_synapse_t*)nimcp_malloc(
        sizeof(triplet_stdp_synapse_t)
    );
    if (!synapse) {
        NIMCP_LOGGING_ERROR("Failed to allocate triplet STDP synapse");
        return NULL;
    }

    /* Initialize fields */
    memset(synapse, 0, sizeof(triplet_stdp_synapse_t));

    /* Weight bounds */
    synapse->w_max = config->w_max;
    synapse->w_min = config->w_min;
    synapse->weight = initial_weight;

    /* Clamp initial weight */
    if (synapse->weight < synapse->w_min) synapse->weight = synapse->w_min;
    if (synapse->weight > synapse->w_max) synapse->weight = synapse->w_max;

    /* Learning parameters */
    synapse->A2_plus = config->A2_plus;
    synapse->A3_plus = config->A3_plus;
    synapse->A2_minus = config->A2_minus;
    synapse->A3_minus = config->A3_minus;

    /* Time constants */
    synapse->tau_plus = config->tau_plus;
    synapse->tau_minus = config->tau_minus;
    synapse->tau_x = config->tau_x;
    synapse->tau_y = config->tau_y;

    /* Initialize traces to zero */
    synapse->r1_pre = 0.0f;
    synapse->r2_pre = 0.0f;
    synapse->o1_post = 0.0f;
    synapse->o2_post = 0.0f;

    /* Initialize spike times */
    synapse->last_pre_spike_time = -1000.0f;  /* Long ago */
    synapse->last_post_spike_time = -1000.0f;

    /* Sleep state */
    synapse->current_sleep_state = SLEEP_STATE_AWAKE;

    /* Statistics */
    synapse->num_ltp_pairwise_events = 0;
    synapse->num_ltp_triplet_events = 0;
    synapse->num_ltd_pairwise_events = 0;
    synapse->num_ltd_triplet_events = 0;
    synapse->total_ltp_pairwise = 0.0f;
    synapse->total_ltp_triplet = 0.0f;
    synapse->total_ltd_pairwise = 0.0f;
    synapse->total_ltd_triplet = 0.0f;

    /* Create mutex for thread safety */
    synapse->mutex = nimcp_platform_mutex_create();
    if (!synapse->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for triplet STDP synapse");
        nimcp_free(synapse);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created triplet STDP synapse (w=%.3f, A2+/A3+/A2-/A3-=%.4f/%.4f/%.4f/%.4f)",
                       initial_weight, config->A2_plus, config->A3_plus,
                       config->A2_minus, config->A3_minus);

    return synapse;
}

void triplet_stdp_synapse_destroy(triplet_stdp_synapse_t* synapse) {
    /* WHAT: Free synapse resources
     * WHY:  Proper cleanup to prevent memory leaks
     * HOW:  Destroy mutex, free structure
     */
    if (!synapse) return;

    /* Destroy mutex */
    if (synapse->mutex) {
        nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)synapse->mutex);
        synapse->mutex = NULL;
    }

    /* Free structure */
    nimcp_free(synapse);

    NIMCP_LOGGING_DEBUG("Destroyed triplet STDP synapse");
}

int triplet_stdp_synapse_reset(triplet_stdp_synapse_t* synapse) {
    /* WHAT: Reset traces and statistics, keep parameters and weight
     * WHY:  Start fresh learning without recreating synapse
     * HOW:  Zero out traces and counters
     */
    if (!synapse) {
        NIMCP_LOGGING_ERROR("NULL synapse in reset");
        return -1;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)synapse->mutex);

    /* Reset traces */
    synapse->r1_pre = 0.0f;
    synapse->r2_pre = 0.0f;
    synapse->o1_post = 0.0f;
    synapse->o2_post = 0.0f;

    /* Reset spike times */
    synapse->last_pre_spike_time = -1000.0f;
    synapse->last_post_spike_time = -1000.0f;

    /* Reset statistics */
    synapse->num_ltp_pairwise_events = 0;
    synapse->num_ltp_triplet_events = 0;
    synapse->num_ltd_pairwise_events = 0;
    synapse->num_ltd_triplet_events = 0;
    synapse->total_ltp_pairwise = 0.0f;
    synapse->total_ltp_triplet = 0.0f;
    synapse->total_ltd_pairwise = 0.0f;
    synapse->total_ltd_triplet = 0.0f;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)synapse->mutex);

    NIMCP_LOGGING_DEBUG("Reset triplet STDP synapse");
    return 0;
}

/* ============================================================================
 * Core Plasticity Functions
 * ============================================================================ */

int triplet_stdp_update_traces(triplet_stdp_synapse_t* synapse, float dt) {
    /* WHAT: Decay all four traces exponentially
     * WHY:  Traces decay between spikes according to time constants
     * HOW:  Multiply each trace by exp(-dt/tau)
     */
    if (!synapse) {
        NIMCP_LOGGING_ERROR("NULL synapse in update_traces");
        return -1;
    }

    if (dt <= 0.0f) return 0;  /* No time passed */

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)synapse->mutex);

    /* Decay fast pre-trace (tau_plus) */
    synapse->r1_pre *= expf(-dt / synapse->tau_plus);

    /* Decay slow pre-trace (tau_x) */
    synapse->r2_pre *= expf(-dt / synapse->tau_x);

    /* Decay fast post-trace (tau_minus) */
    synapse->o1_post *= expf(-dt / synapse->tau_minus);

    /* Decay slow post-trace (tau_y) */
    synapse->o2_post *= expf(-dt / synapse->tau_y);

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)synapse->mutex);

    return 0;
}

float triplet_stdp_pre_spike(triplet_stdp_synapse_t* synapse, float spike_time) {
    /* WHAT: Process presynaptic spike (triggers LTD)
     * WHY:  Pre-before-post → depression
     * HOW:  Compute pairwise + triplet LTD, update traces
     *
     * BIOLOGICAL:
     *   dw_LTD = -A2_minus * o1_post(t_pre) - A3_minus * r1_pre(t_pre) * o2_post(t_pre)
     */
    if (!synapse) {
        NIMCP_LOGGING_ERROR("NULL synapse in pre_spike");
        return 0.0f;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)synapse->mutex);

    /* Update traces if time has passed */
    if (synapse->last_pre_spike_time > 0.0f) {
        float dt = spike_time - synapse->last_pre_spike_time;
        if (dt > 0.0f) {
            synapse->r1_pre *= expf(-dt / synapse->tau_plus);
            synapse->r2_pre *= expf(-dt / synapse->tau_x);
        }
    }

    /* Compute pairwise LTD term */
    float dw_pairwise = -synapse->A2_minus * synapse->o1_post;

    /* Compute triplet LTD term */
    float dw_triplet = -synapse->A3_minus * synapse->r1_pre * synapse->o2_post;

    /* Total weight change */
    float total_dw = dw_pairwise + dw_triplet;

    /* Apply weight change with bounds */
    float old_weight = synapse->weight;
    synapse->weight += total_dw;
    if (synapse->weight < synapse->w_min) synapse->weight = synapse->w_min;
    if (synapse->weight > synapse->w_max) synapse->weight = synapse->w_max;

    float actual_dw = synapse->weight - old_weight;

    /* Update statistics */
    if (dw_pairwise < 0.0f) {
        synapse->num_ltd_pairwise_events++;
        synapse->total_ltd_pairwise += fabsf(dw_pairwise);
    }
    if (dw_triplet < 0.0f) {
        synapse->num_ltd_triplet_events++;
        synapse->total_ltd_triplet += fabsf(dw_triplet);
    }

    /* Increment pre-synaptic traces */
    synapse->r1_pre += 1.0f;
    synapse->r2_pre += 1.0f;

    /* Update last spike time */
    synapse->last_pre_spike_time = spike_time;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)synapse->mutex);

    return actual_dw;
}

float triplet_stdp_post_spike(triplet_stdp_synapse_t* synapse, float spike_time) {
    /* WHAT: Process postsynaptic spike (triggers LTP)
     * WHY:  Post-after-pre → potentiation
     * HOW:  Compute pairwise + triplet LTP, update traces
     *
     * BIOLOGICAL:
     *   dw_LTP = A2_plus * r1_pre(t_post) + A3_plus * r2_pre(t_post) * o1_post(t_post)
     */
    if (!synapse) {
        NIMCP_LOGGING_ERROR("NULL synapse in post_spike");
        return 0.0f;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)synapse->mutex);

    /* Update traces if time has passed */
    if (synapse->last_post_spike_time > 0.0f) {
        float dt = spike_time - synapse->last_post_spike_time;
        if (dt > 0.0f) {
            synapse->o1_post *= expf(-dt / synapse->tau_minus);
            synapse->o2_post *= expf(-dt / synapse->tau_y);
        }
    }

    /* Compute pairwise LTP term */
    float dw_pairwise = synapse->A2_plus * synapse->r1_pre;

    /* Compute triplet LTP term */
    float dw_triplet = synapse->A3_plus * synapse->r2_pre * synapse->o1_post;

    /* Total weight change */
    float total_dw = dw_pairwise + dw_triplet;

    /* Apply weight change with bounds */
    float old_weight = synapse->weight;
    synapse->weight += total_dw;
    if (synapse->weight < synapse->w_min) synapse->weight = synapse->w_min;
    if (synapse->weight > synapse->w_max) synapse->weight = synapse->w_max;

    float actual_dw = synapse->weight - old_weight;

    /* Update statistics */
    if (dw_pairwise > 0.0f) {
        synapse->num_ltp_pairwise_events++;
        synapse->total_ltp_pairwise += dw_pairwise;
    }
    if (dw_triplet > 0.0f) {
        synapse->num_ltp_triplet_events++;
        synapse->total_ltp_triplet += dw_triplet;
    }

    /* Increment post-synaptic traces */
    synapse->o1_post += 1.0f;
    synapse->o2_post += 1.0f;

    /* Update last spike time */
    synapse->last_post_spike_time = spike_time;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)synapse->mutex);

    return actual_dw;
}

/* ============================================================================
 * Sleep Integration
 * ============================================================================ */

int triplet_stdp_set_sleep_state(triplet_stdp_synapse_t* synapse, sleep_state_t state) {
    /* WHAT: Update sleep state for modulation
     * WHY:  Sleep modulates trace time constants
     * HOW:  Store state, will be applied by sleep bridge during updates
     */
    if (!synapse) {
        NIMCP_LOGGING_ERROR("NULL synapse in set_sleep_state");
        return -1;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)synapse->mutex);
    synapse->current_sleep_state = state;
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)synapse->mutex);

    return 0;
}

int triplet_stdp_connect_sleep_bridge(
    triplet_stdp_synapse_t* synapse,
    triplet_stdp_sleep_bridge_t sleep_bridge
) {
    /* WHAT: Link synapse to sleep modulation system
     * WHY:  Enable sleep-dependent plasticity changes
     * HOW:  Register with sleep bridge (implementation in bridge)
     */
    if (!synapse || !sleep_bridge) {
        NIMCP_LOGGING_ERROR("NULL pointer in connect_sleep_bridge");
        return -1;
    }

    /* Bridge will handle registration */
    NIMCP_LOGGING_DEBUG("Connected triplet STDP synapse to sleep bridge");
    return 0;
}

/* ============================================================================
 * Immune Integration
 * ============================================================================ */

int triplet_stdp_connect_immune_bridge(
    triplet_stdp_synapse_t* synapse,
    triplet_stdp_immune_bridge_t immune_bridge
) {
    /* WHAT: Link synapse to immune modulation system
     * WHY:  Enable inflammation-dependent plasticity changes
     * HOW:  Register with immune bridge (implementation in bridge)
     */
    if (!synapse || !immune_bridge) {
        NIMCP_LOGGING_ERROR("NULL pointer in connect_immune_bridge");
        return -1;
    }

    /* Bridge will handle registration */
    NIMCP_LOGGING_DEBUG("Connected triplet STDP synapse to immune bridge");
    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int triplet_stdp_register_callback(
    triplet_stdp_synapse_t* synapse,
    triplet_stdp_callback_t callback,
    void* user_data
) {
    /* WHAT: Register callback for plasticity events
     * WHY:  Allow external monitoring and custom responses
     * HOW:  Store callback, invoke on events (future implementation)
     */
    if (!synapse) {
        NIMCP_LOGGING_ERROR("NULL synapse in register_callback");
        return -1;
    }

    (void)callback;    /* To be implemented */
    (void)user_data;

    NIMCP_LOGGING_DEBUG("Registered callback for triplet STDP events");
    return 0;
}

/* ============================================================================
 * Query Functions
 * ============================================================================ */

float triplet_stdp_get_weight(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->weight;
}

float triplet_stdp_get_r1_pre(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->r1_pre;
}

float triplet_stdp_get_r2_pre(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->r2_pre;
}

float triplet_stdp_get_o1_post(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->o1_post;
}

float triplet_stdp_get_o2_post(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->o2_post;
}

float triplet_stdp_get_total_ltp(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->total_ltp_pairwise + synapse->total_ltp_triplet;
}

float triplet_stdp_get_total_ltd(const triplet_stdp_synapse_t* synapse) {
    if (!synapse) return -1.0f;
    return synapse->total_ltd_pairwise + synapse->total_ltd_triplet;
}

void triplet_stdp_print_stats(const triplet_stdp_synapse_t* synapse) {
    /* WHAT: Display detailed synapse state
     * WHY:  Debugging and monitoring
     * HOW:  Print weight, traces, statistics
     */
    if (!synapse) {
        printf("NULL synapse\n");
        return;
    }

    printf("Triplet STDP Synapse Statistics:\n");
    printf("  Weight: %.4f [%.4f, %.4f]\n", synapse->weight, synapse->w_min, synapse->w_max);
    printf("  Traces: r1=%.4f, r2=%.4f, o1=%.4f, o2=%.4f\n",
           synapse->r1_pre, synapse->r2_pre, synapse->o1_post, synapse->o2_post);
    printf("  LTP Pairwise: %lu events, total=%.6f\n",
           (unsigned long)synapse->num_ltp_pairwise_events, synapse->total_ltp_pairwise);
    printf("  LTP Triplet:  %lu events, total=%.6f\n",
           (unsigned long)synapse->num_ltp_triplet_events, synapse->total_ltp_triplet);
    printf("  LTD Pairwise: %lu events, total=%.6f\n",
           (unsigned long)synapse->num_ltd_pairwise_events, synapse->total_ltd_pairwise);
    printf("  LTD Triplet:  %lu events, total=%.6f\n",
           (unsigned long)synapse->num_ltd_triplet_events, synapse->total_ltd_triplet);
    printf("  Total LTP: %.6f, Total LTD: %.6f\n",
           synapse->total_ltp_pairwise + synapse->total_ltp_triplet,
           synapse->total_ltd_pairwise + synapse->total_ltd_triplet);
    printf("  Net change: %.6f\n",
           (synapse->total_ltp_pairwise + synapse->total_ltp_triplet) -
           (synapse->total_ltd_pairwise + synapse->total_ltd_triplet));
}
