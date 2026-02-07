/**
 * @file nimcp_axon_dendrite_substrate_bridge.c
 * @brief Axon-Dendrite Neural Substrate Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "core/nimcp_axon_dendrite_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>
#include <string.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(axon_dendrite_substrate_bridge)

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Compute Q10 temperature effect
 *
 * WHAT: Calculate rate change from temperature deviation
 * WHY:  Hodgkin-Huxley Q10 coefficient models channel kinetics
 * HOW:  rate(T) = rate(T_ref) × Q10^((T - T_ref)/10)
 *
 * BIOLOGICAL:
 * - Most ion channels: Q10 = 2-3
 * - 10°C change → 2-3x rate change
 * - Hyperthermia → faster kinetics, lower threshold
 * - Hypothermia → slower kinetics, higher threshold
 *
 * @param temperature Current temperature (°C)
 * @param q10 Q10 coefficient (typically 2-3)
 * @return Rate multiplier
 */
static float compute_q10_effect(float temperature, float q10)
{
    if (q10 <= 0.0f) return 1.0f;

    float delta_temp = temperature - SUBSTRATE_REFERENCE_TEMP;
    float exponent = delta_temp / 10.0f;
    float effect = powf(q10, exponent);

    /* Clamp to reasonable range */
    if (effect < SUBSTRATE_VELOCITY_MIN) return SUBSTRATE_VELOCITY_MIN;
    if (effect > SUBSTRATE_VELOCITY_MAX) return SUBSTRATE_VELOCITY_MAX;

    return effect;
}

/**
 * @brief Sigmoid saturation function
 *
 * WHAT: Smooth saturation curve
 * WHY:  Biological systems saturate smoothly, not abruptly
 * HOW:  1 / (1 + exp(-k × (x - x0)))
 *
 * @param x Input value
 * @param x0 Midpoint
 * @param k Steepness
 * @return Saturated output [0-1]
 */
static float sigmoid_saturation(float x, float x0, float k)
{
    float exp_term = expf(-k * (x - x0));
    return 1.0f / (1.0f + exp_term);
}

/**
 * @brief Linear threshold function with saturation
 *
 * WHAT: Linear response above threshold, saturates at 1.0
 * WHY:  Simple model for threshold-dependent effects
 * HOW:  max(0, min(1, (x - threshold) / (1 - threshold)))
 *
 * @param x Input value [0-1]
 * @param threshold Threshold [0-1]
 * @return Response [0-1]
 */
static float linear_threshold(float x, float threshold)
{
    if (x < threshold) return 0.0f;
    if (x >= 1.0f) return 1.0f;

    float range = 1.0f - threshold;
    if (range <= 0.0f) return 1.0f;

    return (x - threshold) / range;
}

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int axon_dendrite_substrate_default_config(axon_dendrite_substrate_config_t* config)
{
    /* WHAT: Provide sensible defaults
     * WHY:  Easy initialization
     * HOW:  Set physiologically-realistic values
     */
    if (!config) {
        NIMCP_LOGGING_ERROR("Null config pointer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_default_config: config is NULL");
        return -1;
    }

    config->enable_axon_modulation = true;
    config->enable_dendrite_modulation = true;
    config->enable_bidirectional_feedback = true;
    config->enable_temperature_effects = true;
    config->enable_atp_dynamics = true;
    config->enable_ion_dynamics = true;
    config->enable_bio_async = false;

    config->temperature_sensitivity = 1.0f;
    config->atp_sensitivity = 1.0f;
    config->ion_sensitivity = 1.0f;
    config->membrane_sensitivity = 1.0f;

    config->min_atp_for_spikes = SUBSTRATE_ATP_THRESHOLD_SPIKE;
    config->min_ion_balance_for_conduct = SUBSTRATE_ION_THRESHOLD_CONDUCT;
    config->min_membrane_for_integration = SUBSTRATE_DENDRITE_MEMBRANE_MIN;

    return 0;
}

axon_dendrite_substrate_bridge_t* axon_dendrite_substrate_bridge_create(
    const axon_dendrite_substrate_config_t* config,
    neural_substrate_t* substrate,
    axon_t* axon,
    dendrite_t* dendrite
)
{
    /* WHAT: Create bridge between substrate and axon/dendrite
     * WHY:  Initialize integration system
     * HOW:  Allocate, validate, connect modules
     */
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Null substrate pointer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;
    }

    if (!axon && !dendrite) {
        NIMCP_LOGGING_ERROR("Must provide at least axon or dendrite");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_bridge_create: required parameter is NULL (axon, dendrite)");
        return NULL;
    }

    axon_dendrite_substrate_bridge_t* bridge =
        (axon_dendrite_substrate_bridge_t*)nimcp_malloc(
            sizeof(axon_dendrite_substrate_bridge_t)
        );
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(axon_dendrite_substrate_bridge_t));

    /* Create mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex");
        nimcp_free(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "axon_dendrite_substrate_bridge_create: bridge->base is NULL");
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex");
        nimcp_free(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "axon_dendrite_substrate_bridge_create: validation failed");
        return NULL;
    }

    /* Store module handles */
    bridge->substrate = substrate;
    bridge->axon = axon;
    bridge->dendrite = dendrite;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        axon_dendrite_substrate_default_config(&bridge->config);
    }

    /* Initialize effects to neutral values */
    bridge->axon_effects.overall_velocity_mod = 1.0f;
    bridge->axon_effects.spike_reliability = 1.0f;
    bridge->axon_effects.refractory_period_mod = 1.0f;
    bridge->axon_effects.transport_efficiency = 1.0f;
    bridge->axon_effects.overall_capacity = 1.0f;

    bridge->dendrite_effects.integration_efficiency = 1.0f;
    bridge->dendrite_effects.spike_threshold_mod = 1.0f;
    bridge->dendrite_effects.plasticity_mod = 1.0f;
    bridge->dendrite_effects.ca_handling_mod = 1.0f;
    bridge->dendrite_effects.overall_capacity = 1.0f;

    bridge->base.bio_async_enabled = false;

    NIMCP_LOGGING_INFO("Created axon-dendrite substrate bridge");

    return bridge;
}

void axon_dendrite_substrate_bridge_destroy(axon_dendrite_substrate_bridge_t* bridge)
{
    /* WHAT: Clean up bridge
     * WHY:  Prevent memory leaks
     * HOW:  Free mutex, structure
     */
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        axon_dendrite_substrate_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Bio-async Integration Implementation
 * ============================================================================ */

int axon_dendrite_substrate_connect_bio_async(axon_dendrite_substrate_bridge_t* bridge)
{
    /* WHAT: Register with bio-async router
     * WHY:  Enable inter-module messaging
     * HOW:  Create module context, register
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_AXON_SUBSTRATE,
        .module_name = "axon_dendrite_substrate_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }

    return 0;
}

int axon_dendrite_substrate_disconnect_bio_async(axon_dendrite_substrate_bridge_t* bridge)
{
    /* WHAT: Unregister from bio-async router
     * WHY:  Clean shutdown
     * HOW:  Unregister module
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    return 0;
}

bool axon_dendrite_substrate_is_bio_async_connected(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_is_bio_async_connected: bridge is NULL");
        return false;
    }
    return bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Substrate → Axon Implementation
 * ============================================================================ */

int axon_dendrite_substrate_update_axon_effects(
    axon_dendrite_substrate_bridge_t* bridge
)
{
    /* WHAT: Compute substrate effects on axon
     * WHY:  Substrate modulates conduction
     * HOW:  Apply Q10, ATP, ion, membrane effects
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!bridge->axon) return 0;  /* No axon to modulate */
    if (!bridge->config.enable_axon_modulation) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get current substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;

    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    axon_substrate_effects_t* effects = &bridge->axon_effects;

    /* 1. Temperature effects (Q10) */
    if (bridge->config.enable_temperature_effects) {
        float temp = physical.temperature * bridge->config.temperature_sensitivity;
        effects->temperature_q10_factor =
            compute_q10_effect(temp, SUBSTRATE_Q10_NA_CHANNEL);
    } else {
        effects->temperature_q10_factor = 1.0f;
    }

    /* 2. ATP effects on velocity and transport */
    if (bridge->config.enable_atp_dynamics) {
        float atp = metabolic.atp_level * bridge->config.atp_sensitivity;

        /* Myelination efficiency requires ATP for Na+/K+-ATPase */
        effects->myelin_efficiency = sigmoid_saturation(
            atp,
            SUBSTRATE_ATP_THRESHOLD_SPIKE,
            10.0f
        );

        /* Velocity scales with ATP availability */
        effects->atp_velocity_factor = linear_threshold(
            atp,
            SUBSTRATE_ATP_THRESHOLD_SPIKE
        );

        /* Transport efficiency (kinesin motors need ATP) */
        effects->transport_efficiency = linear_threshold(
            atp,
            SUBSTRATE_ATP_THRESHOLD_TRANSPORT
        );
        effects->kinesin_activity = effects->transport_efficiency;

        /* Na+/K+-ATPase pump activity */
        effects->pump_activity = sigmoid_saturation(atp, 0.5f, 8.0f);
    } else {
        effects->myelin_efficiency = 1.0f;
        effects->atp_velocity_factor = 1.0f;
        effects->transport_efficiency = 1.0f;
        effects->kinesin_activity = 1.0f;
        effects->pump_activity = 1.0f;
    }

    /* 3. Ion gradient effects */
    if (bridge->config.enable_ion_dynamics) {
        float ion_balance = physical.ion_balance * bridge->config.ion_sensitivity;

        /* Ion gradient strength affects AP amplitude and reliability */
        effects->ion_gradient_strength = ion_balance;
        effects->ap_amplitude_mod =
            0.5f + 0.5f * ion_balance;  /* Range: 0.5-1.0 */

        /* Spike reliability depends on ion gradients */
        effects->spike_reliability = linear_threshold(
            ion_balance,
            SUBSTRATE_ION_THRESHOLD_CONDUCT
        );
    } else {
        effects->ion_gradient_strength = 1.0f;
        effects->ap_amplitude_mod = 1.0f;
        effects->spike_reliability = 1.0f;
    }

    /* 4. Membrane effects */
    float membrane = physical.membrane_integrity * bridge->config.membrane_sensitivity;

    /* Membrane damage increases leak, reduces capacitance */
    effects->membrane_leak_mod = 1.0f +
        (1.0f - membrane) * SUBSTRATE_MEMBRANE_LEAK_FACTOR;
    effects->membrane_capacitance_mod = 0.8f + 0.2f * membrane;

    /* 5. Refractory period modulation */
    /* Faster pump activity → shorter refractory */
    effects->refractory_period_mod =
        1.3f - 0.6f * effects->pump_activity;  /* Range: 0.7-1.3 */

    /* 6. Overall velocity modulation */
    /* Combine temperature, ATP, and myelin effects */
    effects->overall_velocity_mod =
        effects->temperature_q10_factor *
        effects->atp_velocity_factor *
        effects->myelin_efficiency;

    /* Clamp to valid range */
    if (effects->overall_velocity_mod < SUBSTRATE_VELOCITY_MIN) {
        effects->overall_velocity_mod = SUBSTRATE_VELOCITY_MIN;
    }
    if (effects->overall_velocity_mod > SUBSTRATE_VELOCITY_MAX) {
        effects->overall_velocity_mod = SUBSTRATE_VELOCITY_MAX;
    }

    /* 7. Overall axon capacity */
    effects->overall_capacity =
        effects->spike_reliability *
        effects->atp_velocity_factor *
        effects->transport_efficiency;

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

float axon_dendrite_substrate_get_conduction_mod(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->axon_effects.overall_velocity_mod;
}

float axon_dendrite_substrate_get_spike_reliability(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->axon_effects.spike_reliability;
}

float axon_dendrite_substrate_get_refractory_mod(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->axon_effects.refractory_period_mod;
}

/* ============================================================================
 * Substrate → Dendrite Implementation
 * ============================================================================ */

int axon_dendrite_substrate_update_dendrite_effects(
    axon_dendrite_substrate_bridge_t* bridge
)
{
    /* WHAT: Compute substrate effects on dendrite
     * WHY:  Substrate affects integration and plasticity
     * HOW:  Apply cable theory, calcium, plasticity constraints
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!bridge->dendrite) return 0;  /* No dendrite to modulate */
    if (!bridge->config.enable_dendrite_modulation) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get current substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;

    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    dendrite_substrate_effects_t* effects = &bridge->dendrite_effects;

    /* 1. Membrane integrity effects on cable properties */
    float membrane = physical.membrane_integrity * bridge->config.membrane_sensitivity;

    /* R_m decreases with damage (more leak) */
    /* Time constant: τ_m = R_m × C_m */
    effects->membrane_time_constant_mod =
        0.5f + 0.5f * membrane;  /* Range: 0.5-1.0 */

    /* Space constant: λ = √(R_m / R_a) */
    /* Lower R_m → smaller λ → more attenuation */
    effects->space_constant_mod = sqrtf(membrane);

    /* Attenuation increases with membrane damage */
    effects->attenuation_mod = 1.0f + (1.0f - membrane);  /* Range: 1.0-2.0 */

    /* Overall integration efficiency */
    effects->integration_efficiency = linear_threshold(
        membrane,
        bridge->config.min_membrane_for_integration
    );

    /* 2. Ion balance effects on spike threshold */
    if (bridge->config.enable_ion_dynamics) {
        float ion_balance = physical.ion_balance * bridge->config.ion_sensitivity;

        /* Ion imbalance alters NMDA Mg2+ block sensitivity */
        effects->nmda_mg_block_mod = 0.8f + 0.4f * ion_balance;  /* 0.8-1.2 */

        /* Na+ channel availability in dendrites */
        effects->na_channel_availability = ion_balance;

        /* Overall spike threshold shift */
        effects->spike_threshold_mod = effects->nmda_mg_block_mod;
    } else {
        effects->nmda_mg_block_mod = 1.0f;
        effects->na_channel_availability = 1.0f;
        effects->spike_threshold_mod = 1.0f;
    }

    /* 3. Temperature effects on spike threshold */
    if (bridge->config.enable_temperature_effects) {
        float temp = physical.temperature * bridge->config.temperature_sensitivity;
        float temp_effect = compute_q10_effect(temp, SUBSTRATE_Q10_CA_CHANNEL);

        /* Higher temp → lower threshold (faster kinetics) */
        effects->spike_threshold_mod *= (2.0f - temp_effect);  /* Inverse effect */
    }

    /* 4. ATP effects on calcium handling */
    if (bridge->config.enable_atp_dynamics) {
        float atp = metabolic.atp_level * bridge->config.atp_sensitivity;

        /* Ca2+ ATPase pump efficiency */
        effects->ca_pump_efficiency = sigmoid_saturation(
            atp,
            SUBSTRATE_DENDRITE_CA_THRESHOLD,
            8.0f
        );

        /* Buffering capacity (some buffers ATP-dependent) */
        effects->ca_buffer_capacity = 0.7f + 0.3f * atp;

        /* Overall Ca2+ handling */
        effects->ca_handling_mod =
            (effects->ca_pump_efficiency + effects->ca_buffer_capacity) / 2.0f;
    } else {
        effects->ca_pump_efficiency = 1.0f;
        effects->ca_buffer_capacity = 1.0f;
        effects->ca_handling_mod = 1.0f;
    }

    /* 5. Plasticity capacity (ATP-dependent) */
    if (bridge->config.enable_atp_dynamics) {
        float atp = metabolic.atp_level * bridge->config.atp_sensitivity;

        /* LTP requires ATP for spine growth, receptor insertion */
        effects->ltp_capacity = linear_threshold(
            atp,
            SUBSTRATE_DENDRITE_ATP_THRESHOLD
        );

        /* LTD is less ATP-demanding than LTP */
        effects->ltd_capacity = linear_threshold(atp, 0.3f);

        /* Spine structural plasticity */
        effects->spine_growth_capacity = effects->ltp_capacity;

        /* Overall plasticity modulation */
        effects->plasticity_mod =
            (effects->ltp_capacity + effects->ltd_capacity) / 2.0f;
    } else {
        effects->ltp_capacity = 1.0f;
        effects->ltd_capacity = 1.0f;
        effects->spine_growth_capacity = 1.0f;
        effects->plasticity_mod = 1.0f;
    }

    /* 6. Overall dendrite capacity */
    effects->overall_capacity =
        effects->integration_efficiency *
        effects->ca_handling_mod *
        effects->plasticity_mod;

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

float axon_dendrite_substrate_get_integration_mod(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->dendrite_effects.integration_efficiency;
}

float axon_dendrite_substrate_get_spike_threshold_mod(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->dendrite_effects.spike_threshold_mod;
}

float axon_dendrite_substrate_get_plasticity_mod(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->dendrite_effects.plasticity_mod;
}

float axon_dendrite_substrate_get_ca_handling_mod(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge) return 1.0f;
    return bridge->dendrite_effects.ca_handling_mod;
}

/* ============================================================================
 * Axon/Dendrite → Substrate Feedback Implementation
 * ============================================================================ */

int axon_dendrite_substrate_record_axon_spikes(
    axon_dendrite_substrate_bridge_t* bridge,
    uint32_t spike_count
)
{
    /* WHAT: Deplete substrate from axon spikes
     * WHY:  Spikes consume ATP and disrupt ion gradients
     * HOW:  Update substrate state based on spike count
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!bridge->config.enable_bidirectional_feedback) return 0;
    if (spike_count == 0) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Record spikes for statistics */
    bridge->recent_axon_spikes += spike_count;
    bridge->stats.axon_spikes_processed += spike_count;

    /* ATP depletion */
    if (bridge->config.enable_atp_dynamics) {
        float atp_cost = spike_count * SUBSTRATE_ATP_COST_PER_SPIKE;
        bridge->accumulated_atp_debt += atp_cost;

        /* Apply to substrate */
        substrate_record_spikes(bridge->substrate, spike_count);
    }

    /* Ion gradient disruption */
    if (bridge->config.enable_ion_dynamics) {
        float ion_cost = spike_count * SUBSTRATE_ION_ACCUMULATION_RATE;
        bridge->accumulated_ion_imbalance += ion_cost;

        /* Would need setter in substrate API to apply ion disruption */
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int axon_dendrite_substrate_record_dendrite_events(
    axon_dendrite_substrate_bridge_t* bridge,
    uint32_t event_count
)
{
    /* WHAT: Deplete substrate from dendritic events
     * WHY:  Synaptic integration consumes ATP
     * HOW:  Update substrate based on event count
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!bridge->config.enable_bidirectional_feedback) return 0;
    if (event_count == 0) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    bridge->recent_dendrite_events += event_count;
    bridge->stats.dendrite_events_processed += event_count;

    if (bridge->config.enable_atp_dynamics) {
        /* Synaptic events are cheaper than spikes */
        float atp_cost = event_count * SUBSTRATE_COST_PER_TRANSMISSION;
        bridge->accumulated_atp_debt += atp_cost;

        substrate_record_transmissions(bridge->substrate, event_count);
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int axon_dendrite_substrate_record_plasticity(
    axon_dendrite_substrate_bridge_t* bridge,
    float magnitude
)
{
    /* WHAT: Deplete substrate from plasticity
     * WHY:  LTP/LTD is energetically expensive
     * HOW:  Scale ATP cost by plasticity magnitude
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }
    if (!bridge->config.enable_bidirectional_feedback) return 0;
    if (magnitude <= 0.0f) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_atp_dynamics) {
        float atp_cost = magnitude * SUBSTRATE_PLASTICITY_ATP_COST;
        bridge->accumulated_atp_debt += atp_cost;

        /* Plasticity events treated as high-cost transmissions */
        uint32_t equiv_events = (uint32_t)(magnitude * 10.0f);
        substrate_record_transmissions(bridge->substrate, equiv_events);
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int axon_dendrite_substrate_bridge_update(
    axon_dendrite_substrate_bridge_t* bridge,
    uint64_t delta_ms
)
{
    /* WHAT: Process all substrate-axon-dendrite interactions
     * WHY:  Advance coupled state
     * HOW:  Update effects, apply feedback, clear accumulators
     */
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;

    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    bridge->stats.total_updates++;

    /* Update substrate effects on axon */
    if (bridge->axon && bridge->config.enable_axon_modulation) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        axon_dendrite_substrate_update_axon_effects(bridge);
        nimcp_platform_mutex_lock(bridge->base.mutex);
    }

    /* Update substrate effects on dendrite */
    if (bridge->dendrite && bridge->config.enable_dendrite_modulation) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        axon_dendrite_substrate_update_dendrite_effects(bridge);
        nimcp_platform_mutex_lock(bridge->base.mutex);
    }

    /* Check for substrate limitations */
    substrate_metabolic_state_t metabolic;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);

    if (metabolic.atp_level < bridge->config.min_atp_for_spikes) {
        bridge->stats.atp_depletion_events++;
    }

    substrate_physical_state_t physical;
    substrate_get_physical_state(bridge->substrate, &physical);

    if (physical.ion_balance < bridge->config.min_ion_balance_for_conduct) {
        bridge->stats.ion_imbalance_events++;
    }

    /* Update statistics */
    if (bridge->axon_effects.overall_velocity_mod > bridge->stats.peak_velocity_mod) {
        bridge->stats.peak_velocity_mod = bridge->axon_effects.overall_velocity_mod;
    }
    if (bridge->axon_effects.overall_velocity_mod < bridge->stats.min_velocity_mod ||
        bridge->stats.min_velocity_mod == 0.0f) {
        bridge->stats.min_velocity_mod = bridge->axon_effects.overall_velocity_mod;
    }

    /* Running average of integration efficiency */
    float alpha = 0.01f;  /* Smoothing factor */
    bridge->stats.avg_integration_efficiency =
        alpha * bridge->dendrite_effects.integration_efficiency +
        (1.0f - alpha) * bridge->stats.avg_integration_efficiency;

    /* Clear accumulators periodically */
    static uint64_t last_clear = 0;
    if (delta_ms - last_clear > 1000) {  /* Every second */
        bridge->recent_axon_spikes = 0;
        bridge->recent_dendrite_events = 0;
        last_clear = delta_ms;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Query API Implementation
 * ============================================================================ */

int axon_dendrite_substrate_get_axon_effects(
    const axon_dendrite_substrate_bridge_t* bridge,
    axon_substrate_effects_t* effects
)
{
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_get_axon_effects: required parameter is NULL (bridge, effects)");
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->axon_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

int axon_dendrite_substrate_get_dendrite_effects(
    const axon_dendrite_substrate_bridge_t* bridge,
    dendrite_substrate_effects_t* effects
)
{
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_get_dendrite_effects: required parameter is NULL (bridge, effects)");
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *effects = bridge->dendrite_effects;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool axon_dendrite_substrate_is_axon_limited(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    /* WHAT: Check if substrate constraints impair axon
     * WHY:  Detect substrate stress
     * HOW:  Check critical thresholds
     */
    if (!bridge || !bridge->axon) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_is_axon_limited: required parameter is NULL (bridge, bridge->axon)");
        return false;
    }

    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;

    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* Check critical thresholds */
    if (metabolic.atp_level < bridge->config.min_atp_for_spikes) return true;
    if (physical.ion_balance < bridge->config.min_ion_balance_for_conduct) return true;
    if (physical.membrane_integrity < SUBSTRATE_MEMBRANE_THRESHOLD) return true;

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "axon_dendrite_substrate_is_axon_limited: validation failed");
    return false;
}

bool axon_dendrite_substrate_is_dendrite_limited(
    const axon_dendrite_substrate_bridge_t* bridge
)
{
    if (!bridge || !bridge->dendrite) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_is_dendrite_limited: required parameter is NULL (bridge, bridge->dendrite)");
        return false;
    }

    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;

    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    if (metabolic.atp_level < SUBSTRATE_DENDRITE_ATP_THRESHOLD) return true;
    if (physical.membrane_integrity < bridge->config.min_membrane_for_integration) return true;
    if (physical.ion_balance < 0.5f) return true;

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "axon_dendrite_substrate_is_dendrite_limited: validation failed");
    return false;
}

int axon_dendrite_substrate_get_stats(
    const axon_dendrite_substrate_bridge_t* bridge,
    axon_dendrite_substrate_stats_t* stats
)
{
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "axon_dendrite_substrate_get_stats: required parameter is NULL (bridge, stats)");
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    return 0;
}
