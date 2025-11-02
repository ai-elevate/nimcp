/**
 * @file nimcp_neuromodulators.h
 * @brief Neurotransmitter and neuromodulator system for biological realism
 *
 * WHAT: Models 4 major neuromodulatory systems:
 *       - Dopamine (reward, motivation, learning)
 *       - Serotonin (mood, inhibition, patience)
 *       - Acetylcholine (attention, encoding, salience)
 *       - Norepinephrine (arousal, alertness, focus)
 *
 * WHY:  Neuromodulators are critical for:
 *       - Gating plasticity (when to learn)
 *       - Attention allocation (what to process)
 *       - Memory consolidation (what to remember)
 *       - Emotional context (how to value information)
 *
 * HOW:  Two-level architecture:
 *       1. Global: Broadcast neuromodulator levels (volume transmission)
 *       2. Local: Synapse-specific receptor densities (targeted modulation)
 *
 * BIOLOGICAL MAPPING:
 * - Dopamine: VTA/SNc → striatum, cortex (reward prediction error)
 * - Serotonin: Raphe nuclei → widespread (mood, inhibition)
 * - Acetylcholine: Basal forebrain → cortex, hippocampus (attention)
 * - Norepinephrine: Locus coeruleus → cortex (arousal)
 *
 * RAG ETHICS INTEGRATION:
 * - High ethics → dopamine (reward good information)
 * - Low ethics → serotonin (inhibit bad information)
 * - High salience → acetylcholine (attend to important)
 * - Threat detected → norepinephrine (vigilance)
 *
 * DESIGN PATTERNS:
 * - Observer: Neurons observe global neuromodulator changes
 * - Strategy: Different modulation strategies per neurotransmitter
 * - Singleton: Global neuromodulator pool
 * - Mediator: Centralized coordination of modulation
 *
 * PERFORMANCE: O(1) to query levels, O(n) to broadcast updates
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_NEUROMODULATORS_H
#define NIMCP_NEUROMODULATORS_H

#include <stdbool.h>
#include <stdint.h>
#include "nimcp_export.h"
#include "nimcp_neuralnet.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Neurotransmitter Types
//=============================================================================

/**
 * @brief Major neuromodulator systems
 */
typedef enum {
    NEUROMOD_DOPAMINE,      /**< DA: Reward, motivation, learning */
    NEUROMOD_SEROTONIN,     /**< 5-HT: Mood, inhibition, patience */
    NEUROMOD_ACETYLCHOLINE, /**< ACh: Attention, encoding, salience */
    NEUROMOD_NOREPINEPHRINE,/**< NE: Arousal, alertness, stress */
    NEUROMOD_GABA,          /**< GABA: Inhibition (fast) */
    NEUROMOD_GLUTAMATE,     /**< GLU: Excitation (fast) */
    NEUROMOD_COUNT          /**< Total count */
} neuromodulator_type_t;

/**
 * @brief Receptor types for each neuromodulator
 */
typedef enum {
    // Dopamine receptors
    RECEPTOR_D1,  /**< D1: Excitatory, increases plasticity */
    RECEPTOR_D2,  /**< D2: Inhibitory, decreases plasticity */

    // Serotonin receptors
    RECEPTOR_5HT1A, /**< 5-HT1A: Inhibitory, reduces firing */
    RECEPTOR_5HT2A, /**< 5-HT2A: Excitatory, modulates perception */

    // Acetylcholine receptors
    RECEPTOR_NICOTINIC,  /**< Fast excitation */
    RECEPTOR_MUSCARINIC, /**< Slow modulation */

    // Norepinephrine receptors
    RECEPTOR_ALPHA1, /**< α1: Excitatory */
    RECEPTOR_ALPHA2, /**< α2: Inhibitory (auto-receptor) */
    RECEPTOR_BETA,   /**< β: Modulates plasticity */

    RECEPTOR_COUNT
} receptor_type_t;

//=============================================================================
// Neuromodulator Dynamics
//=============================================================================

/**
 * @brief Global neuromodulator concentrations
 *
 * WHAT: System-wide "chemical bath" that affects all neurons
 * WHY:  Models volume transmission - slow, diffuse signaling
 * HOW:  Updated by events (rewards, threats, salience)
 *
 * BIOLOGICAL: Like cerebrospinal fluid concentrations
 * RANGE: All values 0.0 to 1.0 (normalized)
 */
typedef struct {
    float dopamine;      /**< Current DA level (0-1) */
    float serotonin;     /**< Current 5-HT level (0-1) */
    float acetylcholine; /**< Current ACh level (0-1) */
    float norepinephrine;/**< Current NE level (0-1) */

    // Fast neurotransmitters (local, not global)
    float gaba;          /**< GABA tone (0-1) */
    float glutamate;     /**< Glutamate tone (0-1) */

    // Dynamics
    float decay_rates[NEUROMOD_COUNT]; /**< Decay time constants */
    uint64_t last_update;               /**< Last update timestamp */
} neuromodulator_pool_t;

/**
 * @brief Receptor densities per neuron
 *
 * WHAT: How sensitive each neuron is to each neuromodulator
 * WHY:  Different brain regions have different receptor distributions
 * HOW:  Multiplies global concentration by local receptor density
 *
 * BIOLOGICAL: Like receptor expression levels
 * RANGE: 0.0 = no receptors, 1.0 = maximum density
 */
typedef struct {
    float d1_density;         /**< D1 receptor density */
    float d2_density;         /**< D2 receptor density */
    float serotonin_density;  /**< 5-HT receptor density */
    float nicotinic_density;  /**< Nicotinic ACh receptor density */
    float alpha_density;      /**< α-adrenergic receptor density */
    float beta_density;       /**< β-adrenergic receptor density */
} receptor_profile_t;

/**
 * @brief Neuromodulation effects on a synapse
 *
 * WHAT: How neuromodulators change synaptic behavior
 * WHY:  Gates plasticity and transmission strength
 * HOW:  Computed from global levels × local receptors
 */
typedef struct {
    float learning_rate_multiplier;  /**< Scales plasticity (0-2) */
    float transmission_gain;         /**< Scales synaptic strength (0-2) */
    float excitability_shift;        /**< Shifts firing threshold (-0.5 to +0.5) */
    float attention_weight;          /**< Attention modulation (0-1) */
} modulation_effects_t;

//=============================================================================
// Neuromodulator System
//=============================================================================

/**
 * @brief Opaque handle to neuromodulator system
 */
typedef struct neuromodulator_system_struct* neuromodulator_system_t;

/**
 * @brief Configuration for neuromodulator system
 */
typedef struct {
    // Baseline concentrations
    float baseline_dopamine;
    float baseline_serotonin;
    float baseline_acetylcholine;
    float baseline_norepinephrine;

    // Decay time constants (seconds)
    float dopamine_decay;      /**< Default: 2.0s */
    float serotonin_decay;     /**< Default: 10.0s (slow) */
    float acetylcholine_decay; /**< Default: 0.5s (fast) */
    float norepinephrine_decay;/**< Default: 3.0s */

    // Response gains (how much to release per unit stimulus)
    float reward_dopamine_gain;     /**< Default: 0.5 */
    float threat_norepinephrine_gain;/**< Default: 0.7 */
    float salience_acetylcholine_gain;/**< Default: 0.6 */
    float punishment_serotonin_gain; /**< Default: 0.4 */

    // Volume transmission
    bool enable_volume_transmission; /**< Broadcast to all neurons */
    float diffusion_rate;            /**< Spatial spread rate */

} neuromodulator_config_t;

/**
 * @brief Create neuromodulator system
 *
 * @param config System configuration
 * @return Neuromodulator system handle or NULL on error
 */
neuromodulator_system_t neuromodulator_system_create(const neuromodulator_config_t* config);

/**
 * @brief Destroy neuromodulator system
 *
 * @param system System to destroy
 */
void neuromodulator_system_destroy(neuromodulator_system_t system);

/**
 * @brief Get current neuromodulator concentrations
 *
 * @param system Neuromodulator system
 * @param pool Output: current concentrations
 * @return true on success
 */
bool neuromodulator_get_levels(neuromodulator_system_t system, neuromodulator_pool_t* pool);

/**
 * @brief Get current level of a single neuromodulator
 *
 * WHAT: Convenience function to query one neuromodulator concentration
 * WHY:  Simpler API when only one level is needed
 *
 * @param system Neuromodulator system
 * @param type Neuromodulator type to query
 * @return Current concentration (0-1), or 0.0f on error
 */
float neuromodulator_get_level(neuromodulator_system_t system, neuromodulator_type_t type);

/**
 * @brief Set neuromodulator concentration directly
 *
 * @param system Neuromodulator system
 * @param type Neuromodulator type
 * @param level New concentration (0-1)
 * @return true on success
 */
bool neuromodulator_set_level(neuromodulator_system_t system, neuromodulator_type_t type,
                              float level);

/**
 * @brief Update neuromodulator dynamics (decay)
 *
 * WHAT: Applies exponential decay to all concentrations
 * WHY:  Neurotransmitters are cleared by reuptake/metabolism
 * HOW:  c(t+Δt) = c(t) × exp(-Δt/τ)
 *
 * COMPLEXITY: O(1) - fixed number of neuromodulators
 *
 * @param system Neuromodulator system
 * @param dt Time step (seconds)
 * @return true on success
 */
bool neuromodulator_update(neuromodulator_system_t system, float dt);

//=============================================================================
// Event-Driven Neuromodulator Release
//=============================================================================

/**
 * @brief Release dopamine in response to reward
 *
 * WHAT: Reward prediction error → dopamine burst
 * WHY:  Reinforcement learning signal
 * HOW:  δ = reward - prediction, DA ∝ δ
 *
 * BIOLOGICAL: Phasic dopamine from VTA
 *
 * @param system Neuromodulator system
 * @param reward_magnitude Reward size (0-1)
 * @param predicted_reward Expected reward (0-1)
 * @return Actual dopamine released (RPE)
 */
float neuromodulator_release_dopamine(neuromodulator_system_t system, float reward_magnitude,
                                     float predicted_reward);

/**
 * @brief Release serotonin for mood/inhibition
 *
 * WHAT: Punishment/aversion → serotonin increase
 * WHY:  Inhibits impulsive behavior, promotes patience
 * HOW:  5-HT ∝ punishment + uncertainty
 *
 * BIOLOGICAL: Raphe nuclei response to negative outcomes
 *
 * @param system Neuromodulator system
 * @param punishment_magnitude Punishment size (0-1)
 * @return Serotonin released
 */
float neuromodulator_release_serotonin(neuromodulator_system_t system, float punishment_magnitude);

/**
 * @brief Release acetylcholine for attention
 *
 * WHAT: Salience/surprise → acetylcholine burst
 * WHY:  Tags important information for encoding
 * HOW:  ACh ∝ |actual - expected|
 *
 * BIOLOGICAL: Basal forebrain response to salience
 *
 * @param system Neuromodulator system
 * @param salience Importance/surprise (0-1)
 * @return Acetylcholine released
 */
float neuromodulator_release_acetylcholine(neuromodulator_system_t system, float salience);

/**
 * @brief Release norepinephrine for arousal
 *
 * WHAT: Threat/stress → norepinephrine burst
 * WHY:  Increases vigilance and alertness
 * HOW:  NE ∝ threat_level + uncertainty
 *
 * BIOLOGICAL: Locus coeruleus response to threat
 *
 * @param system Neuromodulator system
 * @param threat_level Threat magnitude (0-1)
 * @param uncertainty Uncertainty level (0-1)
 * @return Norepinephrine released
 */
float neuromodulator_release_norepinephrine(neuromodulator_system_t system, float threat_level,
                                           float uncertainty);

//=============================================================================
// Local Receptor-Mediated Effects
//=============================================================================

/**
 * @brief Compute modulation effects for a neuron
 *
 * WHAT: Combines global levels with local receptor densities
 * WHY:  Different neurons respond differently to same modulator
 * HOW:  effect = global_concentration × receptor_density
 *
 * COMPLEXITY: O(1)
 *
 * @param system Neuromodulator system
 * @param receptors Neuron's receptor profile
 * @param effects Output: modulation effects
 * @return true on success
 */
bool neuromodulator_compute_effects(neuromodulator_system_t system,
                                   const receptor_profile_t* receptors,
                                   modulation_effects_t* effects);

/**
 * @brief Apply neuromodulation to learning rate
 *
 * WHAT: Scales plasticity based on neuromodulator levels
 * WHY:  Learning should be gated by context (reward, attention, etc.)
 * HOW:  effective_lr = base_lr × modulation_multiplier
 *
 * ALGORITHM:
 * learning_multiplier =
 *   1.0 +
 *   dopamine × d1_density × 0.5 -        # Enhances learning
 *   dopamine × d2_density × 0.3 +        # Suppresses learning
 *   acetylcholine × nicotinic × 0.4 +   # Enhances encoding
 *   norepinephrine × beta × 0.3         # Enhances consolidation
 *
 * @param base_learning_rate Unmodulated learning rate
 * @param effects Computed modulation effects
 * @return Modulated learning rate
 */
float neuromodulator_modulate_learning_rate(float base_learning_rate,
                                           const modulation_effects_t* effects);

/**
 * @brief Apply neuromodulation to synaptic transmission
 *
 * WHAT: Scales synaptic strength based on neuromodulators
 * WHY:  Attention should amplify relevant signals
 * HOW:  effective_weight = base_weight × transmission_gain
 *
 * @param base_weight Unmodulated synaptic weight
 * @param effects Computed modulation effects
 * @return Modulated synaptic weight
 */
float neuromodulator_modulate_transmission(float base_weight, const modulation_effects_t* effects);

/**
 * @brief Apply neuromodulation to firing threshold
 *
 * WHAT: Shifts neuron excitability based on arousal
 * WHY:  Norepinephrine increases alertness = lower threshold
 * HOW:  effective_threshold = base_threshold + excitability_shift
 *
 * @param base_threshold Unmodulated firing threshold
 * @param effects Computed modulation effects
 * @return Modulated firing threshold
 */
float neuromodulator_modulate_threshold(float base_threshold, const modulation_effects_t* effects);

//=============================================================================
// Receptor Profile Presets
//=============================================================================

/**
 * @brief Create receptor profile for cortical excitatory neuron
 *
 * High D1, moderate ACh, some NE sensitivity
 */
receptor_profile_t neuromodulator_profile_cortical_excitatory(void);

/**
 * @brief Create receptor profile for cortical inhibitory neuron
 *
 * High D2, high GABA, moderate 5-HT
 */
receptor_profile_t neuromodulator_profile_cortical_inhibitory(void);

/**
 * @brief Create receptor profile for hippocampal neuron
 *
 * High ACh (encoding), high D1 (reward), high 5-HT
 */
receptor_profile_t neuromodulator_profile_hippocampal(void);

/**
 * @brief Create receptor profile for striatal neuron
 *
 * Very high dopamine sensitivity (both D1 and D2)
 */
receptor_profile_t neuromodulator_profile_striatal(void);

/**
 * @brief Create receptor profile for amygdala neuron
 *
 * High NE (threat), high dopamine (valence), high 5-HT
 */
receptor_profile_t neuromodulator_profile_amygdala(void);

//=============================================================================
// Integration with Ethics/RAG System
//=============================================================================

/**
 * @brief Compute neuromodulator release from ethics evaluation
 *
 * WHAT: Maps ethics scores to neurotransmitter release
 * WHY:  Ethics = value system, neurotransmitters = value signals
 * HOW:  Direct mapping based on biological principles
 *
 * MAPPING:
 * - High golden_rule_score → dopamine (reward)
 * - Low golden_rule_score → serotonin (inhibit)
 * - High trustworthiness → acetylcholine (attend)
 * - High harm → norepinephrine (threat response)
 *
 * @param system Neuromodulator system
 * @param golden_rule_score Golden rule evaluation (-1 to +1)
 * @param trustworthiness Information trustworthiness (0-1)
 * @param harm_score Harm potential (0-1)
 * @param salience Information salience (0-1)
 * @return true on success
 */
bool neuromodulator_release_from_ethics(neuromodulator_system_t system, float golden_rule_score,
                                       float trustworthiness, float harm_score, float salience);

/**
 * @brief Get effective learning weight from neuromodulators
 *
 * WHAT: Converts neuromodulator state to training weight
 * WHY:  High DA + ACh = strong learning, high 5-HT = weak learning
 * HOW:  Weighted combination of modulator levels
 *
 * ALGORITHM:
 * weight =
 *   0.3 × dopamine +           # Reward
 *   0.3 × acetylcholine +      # Attention
 *   0.2 × (1 - serotonin) +    # Not inhibited
 *   0.2 × norepinephrine       # Aroused
 *
 * @param system Neuromodulator system
 * @param receptors Neuron receptor profile
 * @return Effective learning weight (0-1)
 */
float neuromodulator_get_learning_weight(neuromodulator_system_t system,
                                        const receptor_profile_t* receptors);

//=============================================================================
// Statistics and Monitoring
//=============================================================================

/**
 * @brief Neuromodulator system statistics
 */
typedef struct {
    // Current levels
    float current_dopamine;
    float current_serotonin;
    float current_acetylcholine;
    float current_norepinephrine;

    // Historical averages
    float avg_dopamine;
    float avg_serotonin;
    float avg_acetylcholine;
    float avg_norepinephrine;

    // Release events
    uint64_t dopamine_releases;
    uint64_t serotonin_releases;
    uint64_t acetylcholine_releases;
    uint64_t norepinephrine_releases;

    // Dynamics
    float dopamine_variance;
    float reward_prediction_accuracy;
} neuromodulator_stats_t;

/**
 * @brief Get neuromodulator system statistics
 *
 * @param system Neuromodulator system
 * @param stats Output statistics
 * @return true on success
 */
bool neuromodulator_get_stats(neuromodulator_system_t system, neuromodulator_stats_t* stats);

/**
 * @brief Reset neuromodulator levels to baseline
 *
 * @param system Neuromodulator system
 * @return true on success
 */
bool neuromodulator_reset(neuromodulator_system_t system);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_NEUROMODULATORS_H
