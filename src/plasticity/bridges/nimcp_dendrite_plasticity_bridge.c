/**
 * @file nimcp_dendrite_plasticity_bridge.c
 * @brief Dendrite-Plasticity Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-21
 *
 * WHAT: Connects dendritic compartments to plasticity mechanisms
 * WHY:  Dendrites are primary sites of synaptic plasticity
 * HOW:  Routes calcium signals and STDP events between dendrites and plasticity
 *
 * @author NIMCP Development Team
 */

#include "plasticity/bridges/nimcp_dendrite_plasticity_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

/**
 * @brief Find or create compartment state
 */
static compartment_plasticity_state_t* find_or_create_compartment(
    dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id)
{
    if (!bridge) return NULL;

    /* Search existing compartments */
    for (size_t i = 0; i < bridge->num_compartments; i++) {
        if (bridge->compartments[i].compartment_id == compartment_id) {
            return &bridge->compartments[i];
        }
    }

    /* Create new compartment if capacity allows */
    if (bridge->num_compartments >= bridge->compartment_capacity) {
        return NULL;
    }

    compartment_plasticity_state_t* comp = &bridge->compartments[bridge->num_compartments];
    memset(comp, 0, sizeof(*comp));
    comp->compartment_id = compartment_id;
    comp->bcm_threshold = bridge->config.bcm_sliding_threshold;
    comp->ca_state = CALCIUM_LEVEL_NONE;
    bridge->num_compartments++;

    return comp;
}

/**
 * @brief Classify calcium level
 */
static dendrite_calcium_level_t classify_calcium(
    const dendrite_plasticity_config_t* config,
    float calcium_level)
{
    if (calcium_level >= config->calcium_ltp_threshold) {
        return CALCIUM_LEVEL_LTP;
    }
    if (calcium_level >= config->calcium_ltd_threshold) {
        return CALCIUM_LEVEL_LTD;
    }
    return CALCIUM_LEVEL_NONE;
}

/* NOTE: Message handlers are registered separately via bio_router_register_handler() */

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int dendrite_plasticity_default_config(dendrite_plasticity_config_t* config)
{
    if (!config) return -1;

    memset(config, 0, sizeof(*config));

    /* Calcium thresholds - BCM-style */
    config->calcium_ltp_threshold = 0.7f;
    config->calcium_ltd_threshold = 0.3f;
    config->calcium_decay_tau_ms = 50.0f;

    /* STDP parameters */
    config->stdp_gain = 1.0f;
    config->enable_stdp = true;

    /* Structural plasticity */
    config->spine_growth_threshold = 0.8f;
    config->spine_shrink_threshold = 0.1f;
    config->enable_structural_plasticity = true;

    /* BCM metaplasticity */
    config->bcm_sliding_threshold = 0.5f;
    config->bcm_tau_ms = 1000.0f;
    config->enable_bcm = true;

    /* Bio-async */
    config->enable_bio_async = true;
    config->inbox_capacity = 32;

    return 0;
}

dendrite_plasticity_bridge_t* dendrite_plasticity_create(
    const dendrite_plasticity_config_t* config,
    dendrite_t* dendrite,
    plasticity_orchestrator_t* orch)
{
    /* Allow NULL dendrite/orch for partial initialization */
    dendrite_plasticity_bridge_t* bridge = nimcp_malloc(sizeof(dendrite_plasticity_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate dendrite plasticity bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(*bridge));

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        dendrite_plasticity_default_config(&bridge->config);
    }

    /* Store connections */
    bridge->dendrite = dendrite;
    bridge->plasticity_orch = orch;

    /* Allocate compartment array */
    bridge->compartment_capacity = DENDRITE_PLASTICITY_MAX_COMPARTMENTS;
    bridge->compartments = nimcp_malloc(
        bridge->compartment_capacity * sizeof(compartment_plasticity_state_t));
    if (!bridge->compartments) {
        NIMCP_LOGGING_ERROR("Failed to allocate compartment array");
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->compartments, 0,
           bridge->compartment_capacity * sizeof(compartment_plasticity_state_t));

    /* Allocate mutex */
    bridge->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex");
        nimcp_free(bridge->compartments);
        nimcp_free(bridge);
        return NULL;
    }
    if (nimcp_mutex_init(bridge->mutex, NULL) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex");
        nimcp_free(bridge->mutex);
        nimcp_free(bridge->compartments);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->initialized = true;
    bridge->last_update_time = nimcp_time_get_us();

    NIMCP_LOGGING_INFO("Created dendrite-plasticity bridge");
    return bridge;
}

void dendrite_plasticity_destroy(dendrite_plasticity_bridge_t* bridge)
{
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->bio_async_enabled) {
        dendrite_plasticity_disconnect_bio_async(bridge);
    }

    /* Free mutex */
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free compartment array */
    if (bridge->compartments) {
        nimcp_free(bridge->compartments);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed dendrite-plasticity bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int dendrite_plasticity_connect_stdp(
    dendrite_plasticity_bridge_t* bridge,
    stdp_synapse_t* stdp)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->stdp_template = stdp;
    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Connected STDP to dendrite-plasticity bridge");
    return 0;
}

int dendrite_plasticity_connect_bio_async(dendrite_plasticity_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_CORTICAL_DENDRITIC,
        .module_name = DENDRITE_PLASTICITY_MODULE_NAME,
        .inbox_capacity = bridge->config.inbox_capacity,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected dendrite-plasticity to bio-async");
    }

    return 0;
}

int dendrite_plasticity_disconnect_bio_async(dendrite_plasticity_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->bio_async_enabled) return 0;

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }
    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_INFO("Disconnected dendrite-plasticity from bio-async");
    return 0;
}

bool dendrite_plasticity_is_bio_async_connected(const dendrite_plasticity_bridge_t* bridge)
{
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}

/* ============================================================================
 * Calcium Event API
 * ============================================================================ */

int dendrite_plasticity_update_calcium(
    dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id,
    float calcium_influx)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    compartment_plasticity_state_t* comp = find_or_create_compartment(bridge, compartment_id);
    if (!comp) {
        nimcp_mutex_unlock(bridge->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Update calcium level */
    comp->calcium_level += calcium_influx;
    if (comp->calcium_level > 1.0f) comp->calcium_level = 1.0f;
    if (comp->calcium_level < 0.0f) comp->calcium_level = 0.0f;

    /* Classify calcium state */
    comp->ca_state = classify_calcium(&bridge->config, comp->calcium_level);
    comp->last_calcium_time = nimcp_time_get_us();

    /* Track statistics */
    bridge->stats.calcium_events++;

    /* Trigger plasticity based on calcium level */
    if (comp->ca_state == CALCIUM_LEVEL_LTP) {
        float delta = bridge->config.stdp_gain * 0.1f * comp->calcium_level;
        comp->weight_delta_sum += delta;
        bridge->stats.ltp_events++;
        bridge->stats.total_weight_change += delta;
    } else if (comp->ca_state == CALCIUM_LEVEL_LTD) {
        float delta = -bridge->config.stdp_gain * 0.05f * comp->calcium_level;
        comp->weight_delta_sum += delta;
        bridge->stats.ltd_events++;
        bridge->stats.total_weight_change += delta;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

dendrite_calcium_level_t dendrite_plasticity_get_calcium_state(
    const dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id)
{
    if (!bridge) return CALCIUM_LEVEL_NONE;

    for (size_t i = 0; i < bridge->num_compartments; i++) {
        if (bridge->compartments[i].compartment_id == compartment_id) {
            return bridge->compartments[i].ca_state;
        }
    }
    return CALCIUM_LEVEL_NONE;
}

int dendrite_plasticity_decay_calcium(
    dendrite_plasticity_bridge_t* bridge,
    float dt_ms)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (dt_ms <= 0.0f) return 0;

    nimcp_mutex_lock(bridge->mutex);

    float decay_factor = expf(-dt_ms / bridge->config.calcium_decay_tau_ms);

    for (size_t i = 0; i < bridge->num_compartments; i++) {
        bridge->compartments[i].calcium_level *= decay_factor;
        bridge->compartments[i].ca_state = classify_calcium(
            &bridge->config, bridge->compartments[i].calcium_level);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/* ============================================================================
 * STDP API
 * ============================================================================ */

float dendrite_plasticity_apply_stdp(
    dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id,
    float pre_spike_time,
    float post_spike_time)
{
    if (!bridge) return 0.0f;
    if (!bridge->config.enable_stdp) return 0.0f;

    nimcp_mutex_lock(bridge->mutex);

    compartment_plasticity_state_t* comp = find_or_create_compartment(bridge, compartment_id);
    if (!comp) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0.0f;
    }

    /* Compute STDP window */
    float dt = post_spike_time - pre_spike_time;
    float weight_change = 0.0f;

    /* Classic STDP rule */
    const float tau_plus = 20.0f;  /* LTP time constant */
    const float tau_minus = 20.0f; /* LTD time constant */
    const float a_plus = 0.01f;    /* LTP amplitude */
    const float a_minus = 0.012f;  /* LTD amplitude */

    if (dt > 0) {
        /* Pre before post: LTP */
        weight_change = a_plus * expf(-dt / tau_plus);
    } else if (dt < 0) {
        /* Post before pre: LTD */
        weight_change = -a_minus * expf(dt / tau_minus);
    }

    /* Apply gain and BCM modulation */
    weight_change *= bridge->config.stdp_gain;
    if (bridge->config.enable_bcm && comp->calcium_level < comp->bcm_threshold) {
        weight_change *= -1.0f; /* Invert for BCM below threshold */
    }

    comp->weight_delta_sum += weight_change;
    comp->last_spike_time = (uint64_t)(post_spike_time * 1000.0f);

    bridge->stats.stdp_events++;
    bridge->stats.total_weight_change += weight_change;

    nimcp_mutex_unlock(bridge->mutex);
    return weight_change;
}

int dendrite_plasticity_process_bpap(
    dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id,
    float bpap_time,
    float attenuation)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    compartment_plasticity_state_t* comp = find_or_create_compartment(bridge, compartment_id);
    if (!comp) {
        nimcp_mutex_unlock(bridge->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* BPAP provides calcium signal proportional to amplitude */
    float calcium_from_bpap = (1.0f - attenuation) * 0.3f;
    comp->calcium_level += calcium_from_bpap;
    if (comp->calcium_level > 1.0f) comp->calcium_level = 1.0f;

    comp->last_spike_time = (uint64_t)(bpap_time * 1000.0f);
    comp->activity_trace += 1.0f - attenuation;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/* ============================================================================
 * Structural Plasticity API
 * ============================================================================ */

int dendrite_plasticity_apply_structural(
    dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_structural_plasticity) return 0;

    nimcp_mutex_lock(bridge->mutex);

    compartment_plasticity_state_t* comp = find_or_create_compartment(bridge, compartment_id);
    if (!comp) {
        nimcp_mutex_unlock(bridge->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Check activity trace for structural changes */
    if (comp->activity_trace > bridge->config.spine_growth_threshold) {
        /* Spine growth - would notify dendritic system */
        bridge->stats.structural_events++;
    } else if (comp->activity_trace < bridge->config.spine_shrink_threshold) {
        /* Spine shrinkage */
        bridge->stats.structural_events++;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float dendrite_plasticity_get_spine_density(
    const dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id)
{
    if (!bridge || !bridge->dendrite) return 0.0f;

    /* Would query dendritic system for spine density */
    return 1.0f; /* Default placeholder */
}

/* ============================================================================
 * BCM Metaplasticity API
 * ============================================================================ */

int dendrite_plasticity_update_bcm(
    dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id,
    float activity)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_bcm) return 0;

    nimcp_mutex_lock(bridge->mutex);

    compartment_plasticity_state_t* comp = find_or_create_compartment(bridge, compartment_id);
    if (!comp) {
        nimcp_mutex_unlock(bridge->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Slide threshold toward activity squared (BCM rule) */
    float activity_sq = activity * activity;
    float tau = bridge->config.bcm_tau_ms;
    float alpha = 1.0f / tau; /* Adaptation rate */

    comp->bcm_threshold += alpha * (activity_sq - comp->bcm_threshold);

    /* Clamp threshold */
    if (comp->bcm_threshold < 0.1f) comp->bcm_threshold = 0.1f;
    if (comp->bcm_threshold > 0.9f) comp->bcm_threshold = 0.9f;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float dendrite_plasticity_get_bcm_threshold(
    const dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id)
{
    if (!bridge) return 0.5f;

    for (size_t i = 0; i < bridge->num_compartments; i++) {
        if (bridge->compartments[i].compartment_id == compartment_id) {
            return bridge->compartments[i].bcm_threshold;
        }
    }
    return bridge->config.bcm_sliding_threshold;
}

/* ============================================================================
 * Update and Query API
 * ============================================================================ */

int dendrite_plasticity_update(
    dendrite_plasticity_bridge_t* bridge,
    float dt_ms)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);

    /* Decay calcium */
    float decay_factor = expf(-dt_ms / bridge->config.calcium_decay_tau_ms);
    float sum_calcium = 0.0f;
    float sum_bcm = 0.0f;

    for (size_t i = 0; i < bridge->num_compartments; i++) {
        /* Decay calcium */
        bridge->compartments[i].calcium_level *= decay_factor;
        bridge->compartments[i].ca_state = classify_calcium(
            &bridge->config, bridge->compartments[i].calcium_level);

        /* Decay activity trace */
        bridge->compartments[i].activity_trace *= 0.99f;

        sum_calcium += bridge->compartments[i].calcium_level;
        sum_bcm += bridge->compartments[i].bcm_threshold;
    }

    /* Update averages for stats */
    if (bridge->num_compartments > 0) {
        bridge->stats.avg_calcium_level = sum_calcium / (float)bridge->num_compartments;
        bridge->stats.avg_bcm_threshold = sum_bcm / (float)bridge->num_compartments;
    }

    bridge->last_update_time = nimcp_time_get_us();

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float dendrite_plasticity_get_weight_delta(
    const dendrite_plasticity_bridge_t* bridge,
    uint32_t compartment_id)
{
    if (!bridge) return 0.0f;

    for (size_t i = 0; i < bridge->num_compartments; i++) {
        if (bridge->compartments[i].compartment_id == compartment_id) {
            return bridge->compartments[i].weight_delta_sum;
        }
    }
    return 0.0f;
}

int dendrite_plasticity_apply_to_orchestrator(dendrite_plasticity_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->plasticity_orch) return 0;

    nimcp_mutex_lock(bridge->mutex);

    /* Would send accumulated deltas to orchestrator */
    /* For now, just clear the accumulators */
    for (size_t i = 0; i < bridge->num_compartments; i++) {
        bridge->compartments[i].weight_delta_sum = 0.0f;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dendrite_plasticity_get_stats(
    const dendrite_plasticity_bridge_t* bridge,
    dendrite_plasticity_stats_t* stats)
{
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;

    *stats = bridge->stats;
    return 0;
}

void dendrite_plasticity_reset_stats(dendrite_plasticity_bridge_t* bridge)
{
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->mutex);
}

/* ============================================================================
 * String Conversion
 * ============================================================================ */

const char* dendrite_plasticity_event_to_string(dendrite_plasticity_event_t event)
{
    switch (event) {
        case DENDRITE_EVENT_CALCIUM_SPIKE:   return "CALCIUM_SPIKE";
        case DENDRITE_EVENT_BPAP:            return "BPAP";
        case DENDRITE_EVENT_EPSP:            return "EPSP";
        case DENDRITE_EVENT_SPINE_GROWTH:    return "SPINE_GROWTH";
        case DENDRITE_EVENT_SPINE_SHRINK:    return "SPINE_SHRINK";
        case DENDRITE_EVENT_BRANCH_FORMATION: return "BRANCH_FORMATION";
        case DENDRITE_EVENT_BRANCH_RETRACTION: return "BRANCH_RETRACTION";
        default: return "UNKNOWN";
    }
}

const char* dendrite_calcium_level_to_string(dendrite_calcium_level_t level)
{
    switch (level) {
        case CALCIUM_LEVEL_NONE:    return "NONE";
        case CALCIUM_LEVEL_LTD:     return "LTD";
        case CALCIUM_LEVEL_NEUTRAL: return "NEUTRAL";
        case CALCIUM_LEVEL_LTP:     return "LTP";
        default: return "UNKNOWN";
    }
}
