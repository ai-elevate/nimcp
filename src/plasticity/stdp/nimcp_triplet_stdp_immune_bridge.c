/**
 * @file nimcp_triplet_stdp_immune_bridge.c
 * @brief Triplet STDP-Immune Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-19
 */

#include "plasticity/stdp/nimcp_triplet_stdp_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

int triplet_stdp_immune_default_config(triplet_stdp_immune_config_t* config) {
    /* WHAT: Return default configuration
     * WHY:  Easy initialization with biological defaults
     * HOW:  Set all parameters to Pfister & Gerstner values
     */
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config in default_config");
        return -1;
    }

    config->enable_cytokine_modulation = true;
    config->enable_inflammation_impairment = true;
    config->enable_instability_detection = true;

    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->instability_sensitivity = 1.0f;

    config->base_A2_plus = TRIPLET_STDP_DEFAULT_A2_PLUS;
    config->base_A2_minus = TRIPLET_STDP_DEFAULT_A2_MINUS;
    config->base_A3_plus = TRIPLET_STDP_DEFAULT_A3_PLUS;
    config->base_A3_minus = TRIPLET_STDP_DEFAULT_A3_MINUS;
    config->base_tau_plus = TRIPLET_STDP_DEFAULT_TAU_PLUS;
    config->base_tau_minus = TRIPLET_STDP_DEFAULT_TAU_MINUS;
    config->base_tau_x = TRIPLET_STDP_DEFAULT_TAU_X;
    config->base_tau_y = TRIPLET_STDP_DEFAULT_TAU_Y;

    config->triplet_pairwise_ratio_threshold = TRIPLET_PAIRWISE_RATIO_THRESHOLD;
    config->slow_trace_variance_threshold = SLOW_TRACE_VARIANCE_THRESHOLD;
    config->frequency_selectivity_threshold = FREQUENCY_SELECTIVITY_THRESHOLD;

    return 0;
}

triplet_stdp_immune_bridge_t triplet_stdp_immune_bridge_create(
    const triplet_stdp_immune_config_t* config,
    brain_immune_system_t* immune_system,
    triplet_stdp_synapse_t** synapses,
    size_t num_synapses
) {
    /* WHAT: Create triplet STDP-immune bridge
     * WHY:  Initialize bidirectional integration
     * HOW:  Allocate structure, link subsystems
     */
    if (!immune_system) {
        NIMCP_LOGGING_ERROR("NULL immune_system in bridge create");
        return NULL;
    }

    triplet_stdp_immune_bridge_t bridge =
        (struct triplet_stdp_immune_bridge_struct*)nimcp_malloc(
            sizeof(struct triplet_stdp_immune_bridge_struct)
        );

    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate triplet STDP immune bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(struct triplet_stdp_immune_bridge_struct));

    /* Set configuration */
    triplet_stdp_immune_config_t default_config;
    if (!config) {
        triplet_stdp_immune_default_config(&default_config);
        config = &default_config;
    }

    bridge->immune_system = immune_system;
    bridge->synapses = synapses;
    bridge->num_synapses = num_synapses;
    bridge->synapse_capacity = num_synapses;

    /* Store base parameters */
    bridge->base_A2_plus = config->base_A2_plus;
    bridge->base_A2_minus = config->base_A2_minus;
    bridge->base_A3_plus = config->base_A3_plus;
    bridge->base_A3_minus = config->base_A3_minus;
    bridge->base_tau_plus = config->base_tau_plus;
    bridge->base_tau_minus = config->base_tau_minus;
    bridge->base_tau_x = config->base_tau_x;
    bridge->base_tau_y = config->base_tau_y;

    /* Set flags */
    bridge->enable_cytokine_modulation = config->enable_cytokine_modulation;
    bridge->enable_inflammation_impairment = config->enable_inflammation_impairment;
    bridge->enable_instability_detection = config->enable_instability_detection;

    /* Initialize effects to baseline */
    bridge->cytokine_effects.total_a2_modulation = 1.0f;
    bridge->cytokine_effects.total_a3_modulation = 1.0f;
    bridge->cytokine_effects.total_tau_modulation = 1.0f;

    bridge->inflammation_state.current_level = INFLAMMATION_NONE;
    bridge->inflammation_state.a2_suppression = 0.0f;
    bridge->inflammation_state.a3_suppression = 0.0f;

    /* Create mutex */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_INFO("Triplet STDP immune bridge created (%zu synapses)", num_synapses);
    return bridge;
}

void triplet_stdp_immune_bridge_destroy(triplet_stdp_immune_bridge_t bridge) {
    /* WHAT: Destroy bridge
     * WHY:  Clean up resources
     * HOW:  Destroy mutex, free structure
     */
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy((nimcp_platform_mutex_t*)bridge->mutex);
        bridge->mutex = NULL;
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_DEBUG("Destroyed triplet STDP immune bridge");
}

/* ============================================================================
 * Immune → Triplet STDP Functions
 * ============================================================================ */

int triplet_stdp_immune_apply_cytokine_effects(triplet_stdp_immune_bridge_t bridge) {
    /* WHAT: Apply cytokine modulation to triplet parameters
     * WHY:  Cytokines suppress triplet plasticity
     * HOW:  Query immune system, compute modulation factors
     */
    if (!bridge || !bridge->immune_system) {
        NIMCP_LOGGING_ERROR("NULL pointer in apply_cytokine_effects");
        return -1;
    }

    if (!bridge->enable_cytokine_modulation) return 0;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Get cytokine levels (simplified - would query immune system) */
    float il1_level = 0.0f;
    float il6_level = 0.0f;
    float tnf_level = 0.0f;
    float il10_level = 0.0f;

    /* Compute A2 modulation */
    float a2_factor = 1.0f;
    a2_factor *= (1.0f - il1_level * (1.0f - CYTOKINE_IL1_A2_IMPAIRMENT));
    a2_factor *= (1.0f - il6_level * (1.0f - CYTOKINE_IL6_A2_IMPAIRMENT));
    a2_factor *= (1.0f + il10_level * (CYTOKINE_IL10_RESTORATION - 1.0f));

    /* Compute A3 modulation (more sensitive) */
    float a3_factor = 1.0f;
    a3_factor *= (1.0f - il1_level * (1.0f - CYTOKINE_IL1_A3_IMPAIRMENT));
    a3_factor *= (1.0f - il6_level * (1.0f - CYTOKINE_IL6_A3_IMPAIRMENT));
    a3_factor *= (1.0f + il10_level * (CYTOKINE_IL10_RESTORATION - 1.0f));

    /* Compute tau modulation */
    float tau_factor = 1.0f;
    tau_factor *= (1.0f - tnf_level * (1.0f - CYTOKINE_TNF_TAU_REDUCTION));

    bridge->cytokine_effects.total_a2_modulation = a2_factor;
    bridge->cytokine_effects.total_a3_modulation = a3_factor;
    bridge->cytokine_effects.total_tau_modulation = tau_factor;

    bridge->cytokine_modulations++;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    return 0;
}

int triplet_stdp_immune_apply_inflammation_effects(triplet_stdp_immune_bridge_t bridge) {
    /* WHAT: Apply inflammation-based suppression
     * WHY:  Chronic inflammation reduces triplet plasticity
     * HOW:  Map inflammation level to suppression factors
     */
    if (!bridge || !bridge->immune_system) {
        NIMCP_LOGGING_ERROR("NULL pointer in apply_inflammation_effects");
        return -1;
    }

    if (!bridge->enable_inflammation_impairment) return 0;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Get inflammation level (would query immune system) */
    brain_inflammation_level_t level = bridge->inflammation_state.current_level;

    /* Map to suppression factors */
    float a2_suppression = 0.0f;
    float a3_suppression = 0.0f;

    switch (level) {
        case INFLAMMATION_NONE:
            a2_suppression = 0.0f;
            a3_suppression = 0.0f;
            break;
        case INFLAMMATION_LOCAL:
            a2_suppression = 1.0f - INFLAMMATION_A2_LOCAL;
            a3_suppression = 1.0f - INFLAMMATION_A3_LOCAL;
            break;
        case INFLAMMATION_REGIONAL:
            a2_suppression = 1.0f - INFLAMMATION_A2_REGIONAL;
            a3_suppression = 1.0f - INFLAMMATION_A3_REGIONAL;
            break;
        case INFLAMMATION_SYSTEMIC:
            a2_suppression = 1.0f - INFLAMMATION_A2_SYSTEMIC;
            a3_suppression = 1.0f - INFLAMMATION_A3_SYSTEMIC;
            break;
        case INFLAMMATION_STORM:
            a2_suppression = 1.0f - INFLAMMATION_A2_STORM;
            a3_suppression = 1.0f - INFLAMMATION_A3_STORM;
            break;
    }

    bridge->inflammation_state.a2_suppression = a2_suppression;
    bridge->inflammation_state.a3_suppression = a3_suppression;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    return 0;
}

int triplet_stdp_immune_get_modulation_state(
    const triplet_stdp_immune_bridge_t bridge,
    triplet_stdp_modulation_state_t* modulation
) {
    /* WHAT: Get current modulation state
     * WHY:  Apply to synapses during updates
     * HOW:  Combine cytokine and inflammation effects
     */
    if (!bridge || !modulation) {
        NIMCP_LOGGING_ERROR("NULL pointer in get_modulation_state");
        return -1;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Combine cytokine and inflammation effects */
    float a2_mod = bridge->cytokine_effects.total_a2_modulation *
                   (1.0f - bridge->inflammation_state.a2_suppression);
    float a3_mod = bridge->cytokine_effects.total_a3_modulation *
                   (1.0f - bridge->inflammation_state.a3_suppression);
    float tau_mod = bridge->cytokine_effects.total_tau_modulation;

    modulation->a2_plus_modulation = a2_mod;
    modulation->a2_minus_modulation = a2_mod;
    modulation->a3_plus_modulation = a3_mod;
    modulation->a3_minus_modulation = a3_mod;
    modulation->tau_plus_modulation = tau_mod;
    modulation->tau_minus_modulation = tau_mod;
    modulation->tau_x_modulation = tau_mod;
    modulation->tau_y_modulation = tau_mod;

    /* Compute effective parameters */
    modulation->effective_A2_plus = bridge->base_A2_plus * a2_mod;
    modulation->effective_A2_minus = bridge->base_A2_minus * a2_mod;
    modulation->effective_A3_plus = bridge->base_A3_plus * a3_mod;
    modulation->effective_A3_minus = bridge->base_A3_minus * a3_mod;
    modulation->effective_tau_plus = bridge->base_tau_plus * tau_mod;
    modulation->effective_tau_minus = bridge->base_tau_minus * tau_mod;
    modulation->effective_tau_x = bridge->base_tau_x * tau_mod;
    modulation->effective_tau_y = bridge->base_tau_y * tau_mod;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    return 0;
}

int triplet_stdp_immune_restore_plasticity(
    triplet_stdp_immune_bridge_t bridge,
    float recovery_factor
) {
    /* WHAT: Restore parameters after inflammation resolution
     * WHY:  IL-10 and recovery restore capacity
     * HOW:  Interpolate toward baseline
     */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in restore_plasticity");
        return -1;
    }

    if (recovery_factor < 0.0f) recovery_factor = 0.0f;
    if (recovery_factor > 1.0f) recovery_factor = 1.0f;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)bridge->mutex);

    /* Interpolate suppression toward zero */
    bridge->inflammation_state.a2_suppression *= (1.0f - recovery_factor);
    bridge->inflammation_state.a3_suppression *= (1.0f - recovery_factor);

    bridge->plasticity_restorations++;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)bridge->mutex);

    NIMCP_LOGGING_DEBUG("Restored triplet STDP plasticity (factor=%.2f)", recovery_factor);
    return 0;
}

/* ============================================================================
 * Triplet STDP → Immune Functions
 * ============================================================================ */

int triplet_stdp_immune_detect_instability(triplet_stdp_immune_bridge_t bridge) {
    /* WHAT: Detect abnormal triplet plasticity patterns
     * WHY:  Excessive triplet/pairwise imbalance threatens homeostasis
     * HOW:  Monitor ratios, trace health, frequency selectivity
     */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in detect_instability");
        return -1;
    }

    if (!bridge->enable_instability_detection) return 0;

    /* Would analyze synapse statistics to detect abnormalities */
    /* Simplified implementation */

    return 0;
}

int triplet_stdp_immune_alert_instability(
    triplet_stdp_immune_bridge_t bridge,
    uint32_t* antigen_id
) {
    /* WHAT: Alert immune system of triplet dysfunction
     * WHY:  Abnormal triplet dynamics require immune response
     * HOW:  Create antigen from instability signature
     */
    if (!bridge || !antigen_id) {
        NIMCP_LOGGING_ERROR("NULL pointer in alert_instability");
        return -1;
    }

    /* Would present antigen to immune system */
    *antigen_id = 0;  /* Placeholder */

    bridge->instability_alerts++;

    NIMCP_LOGGING_DEBUG("Alerted immune system of triplet STDP instability");
    return 0;
}

/* ============================================================================
 * Update Function
 * ============================================================================ */

int triplet_stdp_immune_bridge_update(
    triplet_stdp_immune_bridge_t bridge,
    uint64_t delta_ms
) {
    /* WHAT: Update bridge (both directions)
     * WHY:  Process all interactions
     * HOW:  Apply effects, detect instabilities
     */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in update");
        return -1;
    }

    (void)delta_ms;  /* Use for time-based accumulation */

    triplet_stdp_immune_apply_cytokine_effects(bridge);
    triplet_stdp_immune_apply_inflammation_effects(bridge);
    triplet_stdp_immune_detect_instability(bridge);

    bridge->total_updates++;

    return 0;
}

/* ============================================================================
 * Query Functions
 * ============================================================================ */

bool triplet_stdp_immune_is_plasticity_impaired(
    const triplet_stdp_immune_bridge_t bridge
) {
    if (!bridge) return false;

    return (bridge->inflammation_state.a2_suppression > 0.1f ||
            bridge->inflammation_state.a3_suppression > 0.1f);
}

float triplet_stdp_immune_get_triplet_capacity_reduction(
    const triplet_stdp_immune_bridge_t bridge
) {
    if (!bridge) return 0.0f;

    return bridge->inflammation_state.a3_suppression * 100.0f;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int triplet_stdp_immune_connect_bio_async(triplet_stdp_immune_bridge_t bridge) {
    if (!bridge) return -1;

    /* Would register with bio-router */
    bridge->bio_async_enabled = true;

    NIMCP_LOGGING_DEBUG("Connected triplet STDP immune bridge to bio-async");
    return 0;
}

int triplet_stdp_immune_disconnect_bio_async(triplet_stdp_immune_bridge_t bridge) {
    if (!bridge) return -1;

    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_DEBUG("Disconnected triplet STDP immune bridge from bio-async");
    return 0;
}

bool triplet_stdp_immune_is_bio_async_connected(
    const triplet_stdp_immune_bridge_t bridge
) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
