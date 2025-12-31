/**
 * @file nimcp_metabolic_modulation.h
 * @brief Shared metabolic modulation utilities for substrate bridges
 *
 * WHAT: Common utilities for ATP/fatigue-based modulation across substrate bridges
 * WHY: Eliminates code duplication (80+ files had identical clamp_f and modulation code)
 * HOW: Provides nimcp_clamp_f(), metabolic config structs, and generic compute functions
 *
 * This module extracts the common patterns found in substrate bridge implementations:
 * 1. clamp_f() utility function - now nimcp_clamp_f()
 * 2. Metabolic modulation configuration (ATP/fatigue sensitivity, min capacity)
 * 3. Generic metabolic effects computation from substrate state
 */

#ifndef NIMCP_METABOLIC_MODULATION_H
#define NIMCP_METABOLIC_MODULATION_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Clamp a float value between min and max bounds
 *
 * @param value The value to clamp
 * @param min_val Minimum bound (inclusive)
 * @param max_val Maximum bound (inclusive)
 * @return Clamped value
 *
 * Previously duplicated as static clamp_f() in 80+ substrate bridge files.
 */
static inline float nimcp_clamp_f(float value, float min_val, float max_val) {
    return value < min_val ? min_val : (value > max_val ? max_val : value);
}

/**
 * @brief Clamp a double value between min and max bounds
 */
static inline double nimcp_clamp_d(double value, double min_val, double max_val) {
    return value < min_val ? min_val : (value > max_val ? max_val : value);
}

/**
 * @brief Clamp an integer value between min and max bounds
 */
static inline int nimcp_clamp_i(int value, int min_val, int max_val) {
    return value < min_val ? min_val : (value > max_val ? max_val : value);
}

/* ============================================================================
 * Metabolic Modulation Configuration
 * ============================================================================ */

/**
 * @brief Multipliers for metabolic effect computation
 *
 * Replaces the magic numbers (1.1f, 0.9f, 0.95f, 1.05f, etc.) found across
 * substrate bridges. Each bridge can customize these multipliers for its
 * specific cognitive function.
 */
typedef struct metabolic_effect_multipliers {
    float atp_primary_mult;      /**< Multiplier for primary ATP effect (default 1.0) */
    float atp_secondary_mult;    /**< Multiplier for secondary ATP effect (default 1.1) */
    float fatigue_primary_mult;  /**< Multiplier for primary fatigue effect (default 1.0) */
    float fatigue_secondary_mult;/**< Multiplier for secondary fatigue effect (default 0.9) */
} metabolic_effect_multipliers_t;

/**
 * @brief Common metabolic modulation configuration
 *
 * This struct captures the common configuration pattern found in all substrate
 * bridges. Bridge-specific configs can embed this as their first member for
 * easy conversion.
 */
typedef struct metabolic_modulation_config {
    bool enable_atp_modulation;      /**< Enable ATP-based modulation */
    bool enable_fatigue_modulation;  /**< Enable fatigue-based modulation */
    bool enable_bio_async;           /**< Enable bio-async messaging */
    float atp_sensitivity;           /**< ATP sensitivity multiplier (default 1.0) */
    float fatigue_sensitivity;       /**< Fatigue sensitivity multiplier (default 1.0) */
    float min_capacity;              /**< Minimum capacity floor (default 0.2) */
    metabolic_effect_multipliers_t multipliers;  /**< Effect computation multipliers */
} metabolic_modulation_config_t;

/**
 * @brief Generic metabolic effects (4 primary effects + overall capacity)
 *
 * Most substrate bridges compute exactly 4 effects:
 * - 2 ATP-modulated effects (primary_atp, secondary_atp)
 * - 2 fatigue-modulated effects (primary_fatigue, secondary_fatigue)
 * - 1 overall capacity (average of the 4)
 *
 * Bridge-specific effect structs can map their named fields to these generic ones.
 */
typedef struct metabolic_effects {
    float primary_atp;       /**< Primary ATP-modulated effect */
    float secondary_atp;     /**< Secondary ATP-modulated effect */
    float primary_fatigue;   /**< Primary fatigue-modulated effect */
    float secondary_fatigue; /**< Secondary fatigue-modulated effect */
    float overall_capacity;  /**< Average of all 4 effects */
} metabolic_effects_t;

/**
 * @brief Input metabolic state for effect computation
 *
 * Matches the fields extracted from substrate_metabolic_state_t
 */
typedef struct metabolic_input {
    float atp_level;          /**< Current ATP level [0, 1] */
    float metabolic_capacity; /**< Current metabolic capacity [0, 1] */
} metabolic_input_t;

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

/**
 * @brief Get default metabolic modulation configuration
 *
 * Returns configuration matching the existing defaults across bridges:
 * - ATP and fatigue modulation enabled
 * - Bio-async disabled
 * - Sensitivities at 1.0
 * - Min capacity at 0.2
 * - Standard multipliers (1.0, 1.1, 1.0, 0.9)
 */
metabolic_modulation_config_t metabolic_modulation_default_config(void);

/**
 * @brief Get default effect multipliers
 *
 * Returns the standard multipliers:
 * - atp_primary_mult: 1.0
 * - atp_secondary_mult: 1.1
 * - fatigue_primary_mult: 1.0
 * - fatigue_secondary_mult: 0.9
 */
metabolic_effect_multipliers_t metabolic_default_multipliers(void);

/* ============================================================================
 * Effect Computation Functions
 * ============================================================================ */

/**
 * @brief Compute metabolic effects from input state and configuration
 *
 * This replaces the duplicate metabolic modulation code found in all substrate
 * bridge update functions. The pattern is:
 *
 * if (config->enable_atp_modulation) {
 *     effects->primary_atp = clamp_f(atp * config->atp_sensitivity, min, 1.0);
 *     effects->secondary_atp = clamp_f(atp * mult * config->atp_sensitivity, min, 1.0);
 * }
 * if (config->enable_fatigue_modulation) {
 *     effects->primary_fatigue = clamp_f(cap * config->fatigue_sensitivity, min, 1.0);
 *     effects->secondary_fatigue = clamp_f(cap * mult * config->fatigue_sensitivity, min, 1.0);
 * }
 * effects->overall_capacity = average(all 4);
 *
 * @param input Metabolic state input (ATP level and metabolic capacity)
 * @param config Modulation configuration
 * @param effects Output effects structure to populate
 * @return 0 on success, -1 on error
 */
int metabolic_compute_effects(
    const metabolic_input_t* input,
    const metabolic_modulation_config_t* config,
    metabolic_effects_t* effects
);

/**
 * @brief Compute single ATP-modulated effect value
 *
 * Convenience function for bridges that need to compute individual effect values.
 *
 * @param atp_level Current ATP level [0, 1]
 * @param sensitivity ATP sensitivity multiplier
 * @param effect_multiplier Additional multiplier for this specific effect
 * @param min_capacity Minimum capacity floor
 * @return Computed effect value clamped to [min_capacity, 1.0]
 */
float metabolic_compute_atp_effect(
    float atp_level,
    float sensitivity,
    float effect_multiplier,
    float min_capacity
);

/**
 * @brief Compute single fatigue-modulated effect value
 *
 * Convenience function for bridges that need to compute individual effect values.
 *
 * @param metabolic_capacity Current metabolic capacity [0, 1]
 * @param sensitivity Fatigue sensitivity multiplier
 * @param effect_multiplier Additional multiplier for this specific effect
 * @param min_capacity Minimum capacity floor
 * @return Computed effect value clamped to [min_capacity, 1.0]
 */
float metabolic_compute_fatigue_effect(
    float metabolic_capacity,
    float sensitivity,
    float effect_multiplier,
    float min_capacity
);

/**
 * @brief Initialize metabolic effects to full capacity (1.0)
 *
 * Sets all effect fields to 1.0, matching the default initialization
 * pattern in substrate bridge create functions.
 *
 * @param effects Effects structure to initialize
 */
void metabolic_effects_init_full(metabolic_effects_t* effects);

/**
 * @brief Compute overall capacity as average of 4 effects
 *
 * @param effects Effects structure with individual effects populated
 * @return Average of the 4 effect values
 */
float metabolic_compute_overall_capacity(const metabolic_effects_t* effects);

/* ============================================================================
 * Configuration Helpers
 * ============================================================================ */

/**
 * @brief Copy common config fields from bridge-specific config
 *
 * Utility to extract metabolic_modulation_config_t from bridge-specific
 * configuration structs that embed the same fields.
 *
 * @param enable_atp ATP modulation enabled flag
 * @param enable_fatigue Fatigue modulation enabled flag
 * @param enable_bio_async Bio-async enabled flag
 * @param atp_sensitivity ATP sensitivity value
 * @param fatigue_sensitivity Fatigue sensitivity value
 * @param min_capacity Minimum capacity value
 * @param multipliers Optional custom multipliers (NULL for defaults)
 * @return Populated configuration structure
 */
metabolic_modulation_config_t metabolic_config_from_fields(
    bool enable_atp,
    bool enable_fatigue,
    bool enable_bio_async,
    float atp_sensitivity,
    float fatigue_sensitivity,
    float min_capacity,
    const metabolic_effect_multipliers_t* multipliers
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_METABOLIC_MODULATION_H */
