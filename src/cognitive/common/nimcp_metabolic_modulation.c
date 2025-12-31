/**
 * @file nimcp_metabolic_modulation.c
 * @brief Shared metabolic modulation utilities implementation
 *
 * WHAT: Implements common metabolic modulation functions for substrate bridges
 * WHY: Eliminates code duplication across 80+ substrate bridge implementations
 * HOW: Centralizes clamp utilities and metabolic effect computation
 */

#include "cognitive/common/nimcp_metabolic_modulation.h"
#include <stddef.h>

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

metabolic_effect_multipliers_t metabolic_default_multipliers(void) {
    metabolic_effect_multipliers_t mult = {
        .atp_primary_mult = 1.0f,
        .atp_secondary_mult = 1.1f,
        .fatigue_primary_mult = 1.0f,
        .fatigue_secondary_mult = 0.9f
    };
    return mult;
}

metabolic_modulation_config_t metabolic_modulation_default_config(void) {
    metabolic_modulation_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f,
        .multipliers = {
            .atp_primary_mult = 1.0f,
            .atp_secondary_mult = 1.1f,
            .fatigue_primary_mult = 1.0f,
            .fatigue_secondary_mult = 0.9f
        }
    };
    return cfg;
}

/* ============================================================================
 * Effect Computation Functions
 * ============================================================================ */

float metabolic_compute_atp_effect(
    float atp_level,
    float sensitivity,
    float effect_multiplier,
    float min_capacity
) {
    float value = atp_level * effect_multiplier * sensitivity;
    return nimcp_clamp_f(value, min_capacity, 1.0f);
}

float metabolic_compute_fatigue_effect(
    float metabolic_capacity,
    float sensitivity,
    float effect_multiplier,
    float min_capacity
) {
    float value = metabolic_capacity * effect_multiplier * sensitivity;
    return nimcp_clamp_f(value, min_capacity, 1.0f);
}

void metabolic_effects_init_full(metabolic_effects_t* effects) {
    if (!effects) return;

    effects->primary_atp = 1.0f;
    effects->secondary_atp = 1.0f;
    effects->primary_fatigue = 1.0f;
    effects->secondary_fatigue = 1.0f;
    effects->overall_capacity = 1.0f;
}

float metabolic_compute_overall_capacity(const metabolic_effects_t* effects) {
    if (!effects) return 1.0f;

    return (effects->primary_atp + effects->secondary_atp +
            effects->primary_fatigue + effects->secondary_fatigue) / 4.0f;
}

int metabolic_compute_effects(
    const metabolic_input_t* input,
    const metabolic_modulation_config_t* config,
    metabolic_effects_t* effects
) {
    if (!input || !config || !effects) {
        return -1;
    }

    float atp = input->atp_level;
    float cap = input->metabolic_capacity;
    float min = config->min_capacity;

    /* ATP modulation affects primary_atp and secondary_atp */
    if (config->enable_atp_modulation) {
        effects->primary_atp = metabolic_compute_atp_effect(
            atp, config->atp_sensitivity,
            config->multipliers.atp_primary_mult, min
        );
        effects->secondary_atp = metabolic_compute_atp_effect(
            atp, config->atp_sensitivity,
            config->multipliers.atp_secondary_mult, min
        );
    }

    /* Fatigue modulation affects primary_fatigue and secondary_fatigue */
    if (config->enable_fatigue_modulation) {
        effects->primary_fatigue = metabolic_compute_fatigue_effect(
            cap, config->fatigue_sensitivity,
            config->multipliers.fatigue_primary_mult, min
        );
        effects->secondary_fatigue = metabolic_compute_fatigue_effect(
            cap, config->fatigue_sensitivity,
            config->multipliers.fatigue_secondary_mult, min
        );
    }

    /* Compute overall capacity as average */
    effects->overall_capacity = metabolic_compute_overall_capacity(effects);

    return 0;
}

/* ============================================================================
 * Configuration Helpers
 * ============================================================================ */

metabolic_modulation_config_t metabolic_config_from_fields(
    bool enable_atp,
    bool enable_fatigue,
    bool enable_bio_async,
    float atp_sensitivity,
    float fatigue_sensitivity,
    float min_capacity,
    const metabolic_effect_multipliers_t* multipliers
) {
    metabolic_modulation_config_t cfg = {
        .enable_atp_modulation = enable_atp,
        .enable_fatigue_modulation = enable_fatigue,
        .enable_bio_async = enable_bio_async,
        .atp_sensitivity = atp_sensitivity,
        .fatigue_sensitivity = fatigue_sensitivity,
        .min_capacity = min_capacity,
        .multipliers = multipliers ? *multipliers : metabolic_default_multipliers()
    };
    return cfg;
}
