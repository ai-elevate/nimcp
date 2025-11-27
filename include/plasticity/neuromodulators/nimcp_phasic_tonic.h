/**
 * @file nimcp_phasic_tonic.h
 * @brief Phasic vs tonic neuromodulator dynamics (Phase C2.2 Enhancement #2)
 *
 * WHAT: Models burst (phasic) vs baseline (tonic) neurotransmitter release
 * WHY:  Dopamine encodes reward prediction errors via phasic bursts
 * HOW:  Dual-mode concentration with burst triggering and exponential decay
 *
 * BIOLOGICAL BACKGROUND:
 * - Tonic dopamine: 1-5 Hz baseline firing, ~50-100 nM, sustained
 * - Phasic dopamine: 10-20 Hz burst firing, ~1 µM, 100-300ms duration
 * - Phasic bursts signal unexpected rewards (positive TD errors)
 * - Tonic dips signal worse-than-expected outcomes (negative TD errors)
 *
 * @version Phase C2.2 Enhancement #2
 * @date 2025-11-12
 */

#ifndef NIMCP_PHASIC_TONIC_H
#define NIMCP_PHASIC_TONIC_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Phasic-Tonic Configuration and State
// ============================================================================

/**
 * @brief Phasic-tonic neuromodulator system state
 *
 * Models dual-mode release:
 * - Tonic: Sustained baseline concentration (motivation, mood)
 * - Phasic: Transient bursts (learning signals, salience)
 */
typedef struct {
    // === Tonic (Baseline) State ===
    float tonic_level;              /**< Current baseline concentration (µM) */
    float tonic_target;             /**< Homeostatic setpoint (µM) */
    float tonic_min;                /**< Minimum tonic level (µM) */
    float tonic_max;                /**< Maximum tonic level (µM) */
    float homeostatic_tau;          /**< Time constant for homeostatic regulation (seconds) */

    // === Phasic (Burst) State ===
    float phasic_burst;             /**< Current burst amplitude (µM) */
    float burst_decay_tau;          /**< Burst decay time constant (seconds) */
    bool in_burst_state;            /**< Currently in burst mode */
    uint64_t burst_start_time_us;   /**< Burst start timestamp (microseconds) */
    uint32_t burst_duration_ms;     /**< Burst duration (milliseconds) */

    // === Burst Generation Parameters ===
    float burst_amplitude_scale;    /**< Scaling factor for burst amplitude */
    float max_burst_amplitude;      /**< Maximum burst amplitude (µM) */
    float min_burst_amplitude;      /**< Minimum burst amplitude to trigger (µM) */
    float burst_threshold;          /**< Signal threshold for burst triggering */

    // === Autoreceptor Feedback ===
    float autoreceptor_sensitivity; /**< Negative feedback strength [0-1] */
    float feedback_tau;             /**< Feedback time constant (seconds) */

    // === Current Output ===
    float total_concentration;      /**< tonic_level + phasic_burst */
    float release_rate;             /**< Current release rate (µM/s) */

    // === Statistics (for monitoring) ===
    uint32_t burst_count;           /**< Number of bursts triggered */
    uint64_t last_burst_time_us;    /**< Last burst timestamp */
    float avg_inter_burst_interval; /**< Average time between bursts (seconds) */
} phasic_tonic_state_t;

/**
 * @brief Configuration for phasic-tonic system initialization
 */
typedef struct {
    // Tonic defaults
    float initial_tonic;            /**< Initial tonic level (µM) */
    float tonic_target;             /**< Target tonic level (µM) */
    float tonic_range_min;          /**< Minimum tonic (µM) */
    float tonic_range_max;          /**< Maximum tonic (µM) */
    float homeostatic_tau;          /**< Homeostatic time constant (s) */

    // Phasic defaults
    float burst_decay_tau;          /**< Burst decay time constant (s) */
    float burst_amplitude_scale;    /**< Burst scaling factor */
    float max_burst_amplitude;      /**< Max burst (µM) */

    // Autoreceptor
    float autoreceptor_sensitivity; /**< Feedback strength [0-1] */
    float feedback_tau;             /**< Feedback time constant (s) */
} phasic_tonic_config_t;

// ============================================================================
// Biological Parameter Constants
// ============================================================================

/**
 * @brief Dopamine phasic-tonic parameters (from Schultz et al. 2015)
 */
#define DOPAMINE_TONIC_BASELINE     (0.00005f)  /**< 50 nM = 0.00005 µM */
#define DOPAMINE_TONIC_RANGE_MIN    (0.00001f)  /**< 10 nM */
#define DOPAMINE_TONIC_RANGE_MAX    (0.0001f)   /**< 100 nM */
#define DOPAMINE_PHASIC_PEAK        (0.001f)    /**< 1 µM = 1000 nM */
#define DOPAMINE_BURST_DURATION_MS  (200)       /**< 200 ms typical */
#define DOPAMINE_BURST_DECAY_TAU    (0.15f)     /**< 150 ms decay */
#define DOPAMINE_HOMEOSTATIC_TAU    (60.0f)     /**< 60 s regulation */

/**
 * @brief Serotonin phasic-tonic parameters
 */
#define SEROTONIN_TONIC_BASELINE    (0.00003f)  /**< 30 nM */
#define SEROTONIN_BURST_DURATION_MS (500)       /**< Longer bursts */
#define SEROTONIN_BURST_DECAY_TAU   (0.4f)      /**< 400 ms decay */

/**
 * @brief Norepinephrine phasic-tonic parameters
 */
#define NOREPINEPHRINE_TONIC_BASELINE   (0.00002f)  /**< 20 nM */
#define NOREPINEPHRINE_BURST_DURATION_MS (300)       /**< 300 ms */
#define NOREPINEPHRINE_BURST_DECAY_TAU  (0.2f)      /**< 200 ms decay */

// ============================================================================
// Function Declarations
// ============================================================================

/**
 * @brief Create default phasic-tonic configuration for dopamine
 *
 * @return Configuration with biological dopamine parameters
 */
phasic_tonic_config_t phasic_tonic_config_dopamine_default(void);

/**
 * @brief Create default phasic-tonic configuration for serotonin
 *
 * @return Configuration with biological serotonin parameters
 */
phasic_tonic_config_t phasic_tonic_config_serotonin_default(void);

/**
 * @brief Create default phasic-tonic configuration for norepinephrine
 *
 * @return Configuration with biological norepinephrine parameters
 */
phasic_tonic_config_t phasic_tonic_config_norepinephrine_default(void);

/**
 * @brief Initialize phasic-tonic state
 *
 * @param state State structure to initialize
 * @param config Configuration parameters
 * @param current_time_us Current timestamp in microseconds
 */
void phasic_tonic_init(
    phasic_tonic_state_t* state,
    const phasic_tonic_config_t* config,
    uint64_t current_time_us
);

/**
 * @brief Update phasic-tonic dynamics (call every timestep)
 *
 * @param state State to update
 * @param dt Time step (seconds)
 * @param current_time_us Current timestamp (microseconds)
 */
void phasic_tonic_update(
    phasic_tonic_state_t* state,
    float dt,
    uint64_t current_time_us
);

/**
 * @brief Trigger phasic burst (e.g., from TD error)
 *
 * @param state State to update
 * @param amplitude Burst amplitude (typically proportional to TD error)
 * @param duration_ms Burst duration in milliseconds (0 = use default)
 * @param current_time_us Current timestamp
 * @return true if burst triggered, false if suppressed
 */
bool phasic_tonic_trigger_burst(
    phasic_tonic_state_t* state,
    float amplitude,
    uint32_t duration_ms,
    uint64_t current_time_us
);

/**
 * @brief Induce tonic dip (negative reward prediction error)
 *
 * @param state State to update
 * @param magnitude Dip magnitude [0-1] (0 = no effect, 1 = complete suppression)
 */
void phasic_tonic_induce_dip(
    phasic_tonic_state_t* state,
    float magnitude
);

/**
 * @brief Get total neurotransmitter concentration (tonic + phasic)
 *
 * @param state Current state
 * @return Total concentration (µM)
 */
float phasic_tonic_get_concentration(const phasic_tonic_state_t* state);

/**
 * @brief Get current release rate (for integration with vesicle pools)
 *
 * @param state Current state
 * @return Release rate (µM/second)
 */
float phasic_tonic_get_release_rate(const phasic_tonic_state_t* state);

/**
 * @brief Set tonic target (homeostatic setpoint)
 *
 * Allows modeling chronic effects like:
 * - Depression: Lower tonic target
 * - Mania: Higher tonic target
 * - Medication: Adjust target
 *
 * @param state State to modify
 * @param new_target New tonic target (µM)
 */
void phasic_tonic_set_tonic_target(
    phasic_tonic_state_t* state,
    float new_target
);

/**
 * @brief Apply autoreceptor modulation (external regulation)
 *
 * @param state State to modify
 * @param modulation Modulation factor [0-2] (1.0 = no change, <1 = suppress, >1 = enhance)
 */
void phasic_tonic_apply_autoreceptor_modulation(
    phasic_tonic_state_t* state,
    float modulation
);

/**
 * @brief Encode reward prediction error as dopamine dynamics
 *
 * Maps TD error to phasic/tonic changes:
 * - Positive TD error → phasic burst
 * - Negative TD error → tonic dip
 * - Zero TD error → no change
 *
 * @param state Dopamine phasic-tonic state
 * @param td_error Temporal difference error [-1 to +1]
 * @param current_time_us Current timestamp
 * @return true if burst triggered, false otherwise
 */
bool phasic_tonic_encode_td_error(
    phasic_tonic_state_t* state,
    float td_error,
    uint64_t current_time_us
);

/**
 * @brief Get burst statistics for monitoring
 *
 * @param state Current state
 * @param burst_count Output: Number of bursts
 * @param avg_interval Output: Average inter-burst interval (seconds)
 * @param time_since_last Output: Time since last burst (seconds)
 * @param current_time_us Current timestamp
 */
void phasic_tonic_get_burst_statistics(
    const phasic_tonic_state_t* state,
    uint32_t* burst_count,
    float* avg_interval,
    float* time_since_last,
    uint64_t current_time_us
);

/**
 * @brief Reset state to baseline (useful for experiments)
 *
 * @param state State to reset
 * @param current_time_us Current timestamp
 */
void phasic_tonic_reset(
    phasic_tonic_state_t* state,
    uint64_t current_time_us
);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_PHASIC_TONIC_H */
