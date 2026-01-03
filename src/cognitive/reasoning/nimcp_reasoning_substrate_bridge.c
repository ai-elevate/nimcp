/**
 * @file nimcp_reasoning_substrate_bridge.c
 * @brief Reasoning Substrate Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-19
 *
 * WHAT: Integration bridge between reasoning system and neural substrate
 * WHY:  Reasoning processes depend on metabolic resources (ATP, oxygen, glucose)
 * HOW:  Monitors substrate state, computes reasoning modulation, applies effects
 *
 * Uses shared metabolic modulation utilities from nimcp_metabolic_modulation.h
 *
 * BIOLOGICAL BASIS:
 * - Prefrontal cortex (PFC) and parietal networks drive abstract reasoning
 * - Reasoning requires sustained neural firing → high ATP consumption
 * - ATP depletion reduces inference chain depth (shallow thinking)
 * - Metabolic stress impairs working memory → lower logical accuracy
 * - Temperature affects processing speed via Q10 effects
 * - Fatigue slows reasoning and limits abstraction capacity
 */

#include "cognitive/reasoning/nimcp_reasoning_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/memory/nimcp_memory.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions (using shared nimcp_clamp_f from nimcp_metabolic_modulation.h)
 * ============================================================================ */

/**
 * @brief Compute inference depth factor from substrate state
 *
 * WHAT: Calculate maximum reasoning chain length based on ATP and metabolic capacity
 * WHY:  Deep reasoning chains require sustained ATP for continued neural firing
 * HOW:  inference_depth = clamp(atp_level * metabolic_capacity * 1.2, 0.2, 1.0)
 *
 * BIOLOGICAL: Prefrontal networks require sustained activation for multi-step inference.
 *             Low ATP forces shallow, reactive thinking instead of deep reasoning.
 *
 * @param atp_level Current ATP level [0-1]
 * @param metabolic_capacity Metabolic capacity [0-1]
 * @return Inference depth factor [0.2-1.0]
 */
static float compute_inference_depth(float atp_level, float metabolic_capacity)
{
    /* Guard: validate inputs */
    if (atp_level < 0.0f) atp_level = 0.0f;
    if (atp_level > 1.0f) atp_level = 1.0f;
    if (metabolic_capacity < 0.0f) metabolic_capacity = 0.0f;
    if (metabolic_capacity > 1.0f) metabolic_capacity = 1.0f;

    /* Inference depth depends on sustained ATP availability */
    float depth = atp_level * metabolic_capacity * 1.2f;

    /* Clamp to valid range: minimum 0.2 (basic reasoning), maximum 1.0 (full depth) */
    return nimcp_clamp_f(depth, 0.2f, 1.0f);
}

/**
 * @brief Compute logical accuracy factor from substrate state
 *
 * WHAT: Calculate reasoning correctness based on metabolic and physical capacity
 * WHY:  Accurate reasoning requires optimal substrate conditions
 * HOW:  logical_accuracy = clamp((metabolic_capacity + physical_capacity) / 2.0, 0.3, 1.0)
 *
 * BIOLOGICAL: Working memory (critical for logical operations) is highly sensitive
 *             to metabolic stress. Ion imbalance and membrane issues cause errors.
 *
 * @param metabolic_capacity Metabolic capacity [0-1]
 * @param physical_capacity Physical capacity [0-1]
 * @return Logical accuracy factor [0.3-1.0]
 */
static float compute_logical_accuracy(float metabolic_capacity, float physical_capacity)
{
    /* Guard: validate inputs */
    if (metabolic_capacity < 0.0f) metabolic_capacity = 0.0f;
    if (metabolic_capacity > 1.0f) metabolic_capacity = 1.0f;
    if (physical_capacity < 0.0f) physical_capacity = 0.0f;
    if (physical_capacity > 1.0f) physical_capacity = 1.0f;

    /* Accuracy requires both metabolic and physical substrate health */
    float accuracy = (metabolic_capacity + physical_capacity) / 2.0f;

    /* Clamp to valid range: minimum 0.3 (error-prone), maximum 1.0 (highly accurate) */
    return nimcp_clamp_f(accuracy, 0.3f, 1.0f);
}

/**
 * @brief Compute processing speed factor from substrate state
 *
 * WHAT: Calculate reasoning speed based on ATP and temperature
 * WHY:  ATP depletion slows neural firing; temperature affects via Q10
 * HOW:  processing_speed = clamp(atp_level * (1.0 + (temperature - 37.0) * 0.05), 0.3, 1.5)
 *
 * BIOLOGICAL: Q10 ~2-3 for neural processes means 10°C change doubles/halves rate.
 *             ATP depletion reduces firing rate directly.
 *
 * @param atp_level Current ATP level [0-1]
 * @param temperature Current temperature (°C)
 * @return Processing speed factor [0.3-1.5]
 */
static float compute_processing_speed(float atp_level, float temperature)
{
    /* Guard: validate inputs */
    if (atp_level < 0.0f) atp_level = 0.0f;
    if (atp_level > 1.0f) atp_level = 1.0f;
    if (temperature < 20.0f) temperature = 20.0f;  /* Extreme hypothermia */
    if (temperature > 45.0f) temperature = 45.0f;  /* Extreme hyperthermia */

    /* Temperature effect: approximately 5% change per degree from 37°C */
    float temp_factor = 1.0f + (temperature - 37.0f) * 0.05f;

    /* Processing speed scales with ATP and temperature */
    float speed = atp_level * temp_factor;

    /* Clamp to valid range: minimum 0.3 (very slow), maximum 1.5 (enhanced) */
    return nimcp_clamp_f(speed, 0.3f, 1.5f);
}

/**
 * @brief Compute abstraction capacity factor from substrate state
 *
 * WHAT: Calculate ability to form abstract concepts from metabolic capacity
 * WHY:  High-level abstraction is metabolically expensive
 * HOW:  abstraction_capacity = clamp(metabolic_capacity * 0.9, 0.2, 1.0)
 *
 * BIOLOGICAL: Abstraction requires coordinated PFC-parietal activity.
 *             Metabolic stress forces concrete, stimulus-bound thinking.
 *
 * @param metabolic_capacity Metabolic capacity [0-1]
 * @return Abstraction capacity factor [0.2-1.0]
 */
static float compute_abstraction_capacity(float metabolic_capacity)
{
    /* Guard: validate inputs */
    if (metabolic_capacity < 0.0f) metabolic_capacity = 0.0f;
    if (metabolic_capacity > 1.0f) metabolic_capacity = 1.0f;

    /* Abstraction capacity is 90% of metabolic capacity (expensive process) */
    float capacity = metabolic_capacity * 0.9f;

    /* Clamp to valid range: minimum 0.2 (concrete only), maximum 1.0 (full abstraction) */
    return nimcp_clamp_f(capacity, 0.2f, 1.0f);
}

/**
 * @brief Determine if reasoning is impaired based on effects
 *
 * WHAT: Check if reasoning is severely degraded
 * WHY:  Critical failures need explicit flagging
 * HOW:  is_impaired = (inference_depth < 0.6 || logical_accuracy < 0.7)
 *
 * BIOLOGICAL: Models executive dysfunction threshold
 *
 * @param inference_depth Inference depth factor [0-1]
 * @param logical_accuracy Logical accuracy factor [0-1]
 * @return true if impaired, false otherwise
 */
static bool check_impairment(float inference_depth, float logical_accuracy)
{
    /* Impairment if inference depth below 0.6 or accuracy below 0.7 */
    return (inference_depth < 0.6f || logical_accuracy < 0.7f);
}

/**
 * @brief Update running average for statistics
 *
 * WHAT: Exponential moving average update
 * WHY:  Track average values over time
 * HOW:  new_avg = old_avg * 0.95 + new_value * 0.05
 *
 * @param current_avg Current average
 * @param new_value New value to incorporate
 * @return Updated average
 */
static float update_running_average(float current_avg, float new_value)
{
    return current_avg * 0.95f + new_value * 0.05f;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

void reasoning_substrate_default_config(reasoning_substrate_config_t* config)
{
    /* Guard: validate config pointer */
    if (!config) {
        NIMCP_LOGGING_ERROR("Cannot set default config: NULL config");
        return;
    }

    /* Enable all modulations by default */
    config->enable_atp_modulation = true;
    config->enable_fatigue_modulation = true;
    config->enable_stress_modulation = true;

    /* Default sensitivity: 1.0 = normal biological response */
    config->atp_sensitivity = 1.0f;
    config->fatigue_sensitivity = 1.0f;
    config->stress_sensitivity = 1.0f;

    /* Update every 100ms by default */
    config->update_interval_ms = 100;

    /* Enable bio-async messaging */
    config->enable_bio_async = true;

    NIMCP_LOGGING_DEBUG("Reasoning substrate bridge config initialized to defaults");
}

reasoning_substrate_bridge_t* reasoning_substrate_bridge_create(
    const reasoning_substrate_config_t* config,
    nimcp_reasoning_system_t* reasoning,
    neural_substrate_t* substrate
)
{
    /* Guard: validate required pointers */
    if (!reasoning) {
        NIMCP_LOGGING_ERROR("Cannot create bridge: NULL reasoning system");
        return NULL;
    }
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Cannot create bridge: NULL substrate");
        return NULL;
    }

    /* Allocate bridge structure */
    reasoning_substrate_bridge_t* bridge =
        (reasoning_substrate_bridge_t*)nimcp_malloc(sizeof(reasoning_substrate_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate reasoning substrate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(reasoning_substrate_bridge_t));

    /* Store system handles */
    bridge->substrate = substrate;
    bridge->reasoning = reasoning;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        reasoning_substrate_default_config(&bridge->config);
    }

    /* Create mutex */
    bridge->base.mutex = (nimcp_platform_mutex_t*)nimcp_malloc(sizeof(nimcp_platform_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize effects to optimal values */
    bridge->effects.inference_depth = 1.0f;
    bridge->effects.logical_accuracy = 1.0f;
    bridge->effects.processing_speed = 1.0f;
    bridge->effects.abstraction_capacity = 1.0f;
    bridge->effects.is_impaired = false;

    /* Initialize statistics */
    bridge->stats.update_count = 0;
    bridge->stats.impairment_count = 0;
    bridge->stats.low_atp_count = 0;
    bridge->stats.high_fatigue_count = 0;
    bridge->stats.high_stress_count = 0;
    bridge->stats.avg_inference_depth = 1.0f;
    bridge->stats.avg_logical_accuracy = 1.0f;
    bridge->stats.avg_processing_speed = 1.0f;
    bridge->stats.min_atp_observed = 1.0f;
    bridge->stats.max_fatigue_observed = 0.0f;
    bridge->stats.max_stress_observed = 0.0f;

    /* Initialize timestamp */
    bridge->last_update_ms = 0;

    /* Mark as active */
    bridge->is_active = true;

    /* Connect bio-async if enabled */
    if (bridge->config.enable_bio_async) {
        reasoning_substrate_connect_bio_async(bridge);
    }

    NIMCP_LOGGING_INFO("Reasoning substrate bridge created");
    return bridge;
}

void reasoning_substrate_bridge_destroy(reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        return;
    }

    /* Disconnect bio-async */
    if (bridge->base.bio_async_enabled) {
        reasoning_substrate_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Reasoning substrate bridge destroyed");
}

/* ============================================================================
 * Bio-async Integration
 * ============================================================================ */

int reasoning_substrate_connect_bio_async(reasoning_substrate_bridge_t* bridge)
{
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot connect bio-async: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: already connected */
    if (bridge->base.bio_async_enabled) {
        return NIMCP_SUCCESS;
    }

    /* Register with bio-async router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_REASONING,
        .module_name = "reasoning_substrate_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Reasoning substrate bridge connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }

    return NIMCP_SUCCESS;
}

int reasoning_substrate_disconnect_bio_async(reasoning_substrate_bridge_t* bridge)
{
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot disconnect bio-async: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: not connected */
    if (!bridge->base.bio_async_enabled) {
        return NIMCP_SUCCESS;
    }

    /* Unregister from bio-async router */
    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Reasoning substrate bridge disconnected from bio-async router");
    return NIMCP_SUCCESS;
}

bool reasoning_substrate_is_bio_async_connected(const reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        return false;
    }
    return bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int reasoning_substrate_update(reasoning_substrate_bridge_t* bridge)
{
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot update: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if substrate connected */
    if (!bridge->substrate) {
        NIMCP_LOGGING_ERROR("Cannot update: NULL substrate");
        return NIMCP_ERROR_INVALID_STATE;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Query substrate metabolic state */
    substrate_metabolic_state_t metabolic;
    int ret = substrate_get_metabolic_state(bridge->substrate, &metabolic);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate metabolic state");
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return ret;
    }

    /* Query substrate physical state */
    substrate_physical_state_t physical;
    ret = substrate_get_physical_state(bridge->substrate, &physical);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate physical state");
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return ret;
    }

    /* Compute inference depth factor */
    if (bridge->config.enable_atp_modulation) {
        float atp_factor = metabolic.atp_level * bridge->config.atp_sensitivity;
        bridge->effects.inference_depth = compute_inference_depth(
            atp_factor, metabolic.metabolic_capacity);
    } else {
        bridge->effects.inference_depth = 1.0f;
    }

    /* Compute logical accuracy factor */
    if (bridge->config.enable_stress_modulation) {
        float stress_factor = bridge->config.stress_sensitivity;
        bridge->effects.logical_accuracy = compute_logical_accuracy(
            metabolic.metabolic_capacity * stress_factor,
            physical.physical_capacity * stress_factor);
    } else {
        bridge->effects.logical_accuracy = 1.0f;
    }

    /* Compute processing speed factor */
    if (bridge->config.enable_atp_modulation) {
        float atp_factor = metabolic.atp_level * bridge->config.atp_sensitivity;
        bridge->effects.processing_speed = compute_processing_speed(
            atp_factor, physical.temperature);
    } else {
        bridge->effects.processing_speed = 1.0f;
    }

    /* Compute abstraction capacity factor */
    if (bridge->config.enable_stress_modulation) {
        float stress_factor = bridge->config.stress_sensitivity;
        bridge->effects.abstraction_capacity = compute_abstraction_capacity(
            metabolic.metabolic_capacity * stress_factor);
    } else {
        bridge->effects.abstraction_capacity = 1.0f;
    }

    /* Determine impairment status */
    bridge->effects.is_impaired = check_impairment(
        bridge->effects.inference_depth,
        bridge->effects.logical_accuracy);

    /* Update statistics */
    bridge->stats.update_count++;

    if (bridge->effects.is_impaired) {
        bridge->stats.impairment_count++;
    }

    if (metabolic.atp_level < REASONING_ATP_OPTIMAL_THRESHOLD) {
        bridge->stats.low_atp_count++;
    }

    /* Track fatigue (using metabolic_rate as proxy) */
    float fatigue = 1.0f - metabolic.metabolic_capacity;
    if (fatigue > REASONING_FATIGUE_MODERATE_THRESHOLD) {
        bridge->stats.high_fatigue_count++;
    }

    /* Track stress (using physical_capacity degradation as proxy) */
    float stress = 1.0f - physical.physical_capacity;
    if (stress > REASONING_STRESS_MODERATE_THRESHOLD) {
        bridge->stats.high_stress_count++;
    }

    /* Update running averages */
    bridge->stats.avg_inference_depth = update_running_average(
        bridge->stats.avg_inference_depth, bridge->effects.inference_depth);
    bridge->stats.avg_logical_accuracy = update_running_average(
        bridge->stats.avg_logical_accuracy, bridge->effects.logical_accuracy);
    bridge->stats.avg_processing_speed = update_running_average(
        bridge->stats.avg_processing_speed, bridge->effects.processing_speed);

    /* Track extremes */
    if (metabolic.atp_level < bridge->stats.min_atp_observed) {
        bridge->stats.min_atp_observed = metabolic.atp_level;
    }
    if (fatigue > bridge->stats.max_fatigue_observed) {
        bridge->stats.max_fatigue_observed = fatigue;
    }
    if (stress > bridge->stats.max_stress_observed) {
        bridge->stats.max_stress_observed = stress;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Reasoning substrate updated: inference_depth=%.3f, accuracy=%.3f, speed=%.3f, abstraction=%.3f, impaired=%d",
        bridge->effects.inference_depth,
        bridge->effects.logical_accuracy,
        bridge->effects.processing_speed,
        bridge->effects.abstraction_capacity,
        bridge->effects.is_impaired);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

float reasoning_substrate_get_inference_depth(const reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get inference depth: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.inference_depth;
}

float reasoning_substrate_get_logical_accuracy(const reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get logical accuracy: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.logical_accuracy;
}

float reasoning_substrate_get_processing_speed(const reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get processing speed: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.processing_speed;
}

float reasoning_substrate_get_abstraction_capacity(const reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get abstraction capacity: NULL bridge");
        return -1.0f;
    }

    return bridge->effects.abstraction_capacity;
}

const reasoning_substrate_effects_t* reasoning_substrate_get_effects(
    const reasoning_substrate_bridge_t* bridge
)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get effects: NULL bridge");
        return NULL;
    }

    return &bridge->effects;
}

bool reasoning_substrate_is_impaired(const reasoning_substrate_bridge_t* bridge)
{
    if (!bridge) {
        return false;
    }

    return bridge->effects.is_impaired;
}

const reasoning_substrate_stats_t* reasoning_substrate_get_stats(
    const reasoning_substrate_bridge_t* bridge
)
{
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get stats: NULL bridge");
        return NULL;
    }

    return &bridge->stats;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int reasoning_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Reasoning_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Reasoning_Substrate_Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Reasoning_Substrate_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Reasoning_Substrate_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
