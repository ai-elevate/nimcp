/**
 * @file nimcp_neuron_types.h
 * @brief Extended neuron type specialization system
 *
 * DESIGN PHILOSOPHY:
 * - Backward compatible: Existing E/I types continue to work
 * - Non-invasive: Optional type-specific parameters via union
 * - Performance: Type-specific optimized dynamics
 * - Biological realism: Models real neuron diversity
 *
 * NEURON TYPE HIERARCHY:
 * 1. Basic: EXCITATORY, INHIBITORY (existing)
 * 2. Visual: Edge detectors, orientation-selective, color-opponent
 * 3. Auditory: Frequency-tuned, onset detectors, ITD-sensitive
 * 4. Motor: Pattern generators, motoneurons, Renshaw cells
 * 5. Interneuron: Fast-spiking, adaptive, chandelier
 *
 * USAGE EXAMPLE:
 * ```c
 * // Create visual edge detector neuron
 * neuron_type_params_t params = {
 *     .visual_edge = {
 *         .orientation = 45.0f,      // 45-degree edge
 *         .spatial_frequency = 2.0f,  // cycles per degree
 *         .on_center = true
 *     }
 * };
 *
 * neuron_t* n = neuron_create_specialized(id, NEURON_VISUAL_EDGE, &params);
 * ```
 *
 * TDD STATUS: Header-first design, tests to be written
 */

#ifndef NIMCP_NEURON_TYPES_H
#define NIMCP_NEURON_TYPES_H

#include "utils/validation/nimcp_common.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// EXTENDED NEURON TYPES (Backward compatible with existing E/I)
// ============================================================================

/**
 * @brief Extended neuron types for task-specific processing
 *
 * BACKWARD COMPATIBILITY:
 * - Values 0-1 match existing NEURON_EXCITATORY/INHIBITORY
 * - New types start at 100+ to avoid conflicts
 */
typedef enum {
    // ========== VISUAL CORTEX NEURONS (100-199) ==========
    // Note: Values 0-1 (NEURON_EXCITATORY/INHIBITORY) defined in nimcp_neuralnet.h
    // Extended types start at 100 to avoid conflicts
    NEURON_VISUAL_EDGE = 100,     /**< Edge detector (V1 simple cell) */
    NEURON_VISUAL_ORIENTATION,    /**< Orientation-selective (V1 complex) */
    NEURON_VISUAL_DIRECTION,      /**< Direction-selective (MT/V5) */
    NEURON_VISUAL_COLOR_OPPONENT, /**< Color-opponent cell (V4) */
    NEURON_VISUAL_RECEPTIVE_FIELD,/**< Gabor filter receptive field */

    // ========== AUDITORY CORTEX NEURONS (200-299) ==========
    NEURON_AUDITORY_FREQUENCY = 200,  /**< Frequency-tuned (A1) */
    NEURON_AUDITORY_ONSET,            /**< Onset detector */
    NEURON_AUDITORY_ITD,              /**< Interaural time difference */
    NEURON_AUDITORY_ILD,              /**< Interaural level difference */
    NEURON_AUDITORY_COMPLEX,          /**< Complex spectrotemporal */

    // ========== MOTOR NEURONS (300-399) ==========
    NEURON_MOTOR_ALPHA = 300,     /**< Alpha motoneuron */
    NEURON_MOTOR_GAMMA,           /**< Gamma motoneuron */
    NEURON_MOTOR_PATTERN_GEN,     /**< Central pattern generator */
    NEURON_MOTOR_RENSHAW,         /**< Renshaw inhibitory cell */
    NEURON_MOTOR_PROPRIOCEPTIVE,  /**< Proprioceptive feedback */

    // ========== INTERNEURONS (400-499) ==========
    NEURON_FAST_SPIKING = 400,    /**< Fast-spiking interneuron (PV+) */
    NEURON_ADAPTIVE_SPIKING,      /**< Adaptive spiking interneuron */
    NEURON_BURST_SPIKING,         /**< Bursting neuron */
    NEURON_CHANDELIER,            /**< Chandelier cell (axo-axonic) */
    NEURON_BASKET,                /**< Basket cell (soma-targeting) */

    // ========== CORTICAL PYRAMIDAL (500-599) ==========
    NEURON_PYRAMIDAL_L23 = 500,   /**< Layer 2/3 pyramidal */
    NEURON_PYRAMIDAL_L5_THICK,    /**< Layer 5 thick-tufted */
    NEURON_PYRAMIDAL_L5_THIN,     /**< Layer 5 thin-tufted */
    NEURON_PYRAMIDAL_L6,          /**< Layer 6 corticothalamic */

    // ========== SENSORY NEURONS (600-699) ==========
    NEURON_SENSORY_MECHANORECEPTOR = 600, /**< Touch receptor */
    NEURON_SENSORY_NOCICEPTOR,            /**< Pain receptor */
    NEURON_SENSORY_THERMORECEPTOR,        /**< Temperature receptor */
    NEURON_SENSORY_PHOTORECEPTOR,         /**< Light receptor (rod/cone) */

    NEURON_TYPE_COUNT  /**< Total number of neuron types */
} neuron_type_extended_t;

// ============================================================================
// TYPE-SPECIFIC PARAMETER STRUCTURES
// ============================================================================

/**
 * @brief Visual edge detector parameters (Gabor filter)
 */
typedef struct {
    float orientation;        /**< Preferred orientation (degrees, 0-180) */
    float spatial_frequency;  /**< Spatial frequency (cycles per degree) */
    float phase;             /**< Phase offset (0-2π) */
    float aspect_ratio;      /**< Aspect ratio of Gabor (typically 0.5) */
    bool on_center;          /**< ON-center vs OFF-center */
    float receptive_field_size; /**< Size of receptive field (degrees) */
} visual_edge_params_t;

/**
 * @brief Orientation-selective neuron parameters
 */
typedef struct {
    float preferred_orientation; /**< Preferred orientation (degrees) */
    float orientation_bandwidth; /**< Tuning bandwidth (degrees) */
    float direction_selectivity; /**< 0=non-directional, 1=fully directional */
    float surround_suppression;  /**< Surround suppression strength */
} visual_orientation_params_t;

/**
 * @brief Auditory frequency-tuned neuron parameters
 */
typedef struct {
    float center_frequency;      /**< Center frequency (Hz) */
    float bandwidth;             /**< Frequency bandwidth (octaves) */
    float q_factor;              /**< Quality factor (sharpness) */
    float temporal_integration;  /**< Integration window (ms) */
    float adaptation_rate;       /**< Adaptation to sustained input */
} auditory_frequency_params_t;

/**
 * @brief Onset detector parameters
 */
typedef struct {
    float onset_threshold;       /**< Threshold for onset detection */
    float refractory_duration;   /**< Refractory after onset (ms) */
    float decay_time_constant;   /**< Decay after onset (ms) */
    bool offset_sensitive;       /**< Also detect offsets */
} auditory_onset_params_t;

/**
 * @brief Motor pattern generator parameters
 */
typedef struct {
    float rhythm_frequency;      /**< Intrinsic rhythm (Hz) */
    float burst_duration;        /**< Burst duration (ms) */
    float interburst_interval;   /**< Time between bursts (ms) */
    float phase_offset;          /**< Phase relative to other CPGs */
    bool pacemaker;              /**< Is this a pacemaker neuron? */
} motor_pattern_params_t;

/**
 * @brief Fast-spiking interneuron parameters
 */
typedef struct {
    float max_firing_rate;       /**< Maximum sustained rate (Hz) */
    float membrane_time_constant;/**< Fast membrane dynamics (ms) */
    float spike_width;           /**< Narrow spike width (ms) */
    float afterhyperpolarization;/**< AHP amplitude (mV) */
} fast_spiking_params_t;

/**
 * @brief Bursting neuron parameters
 */
typedef struct {
    float burst_threshold;       /**< Threshold for burst initiation */
    float burst_duration;        /**< Duration of burst (ms) */
    float spikes_per_burst;      /**< Number of spikes in burst */
    float interburst_interval;   /**< Minimum time between bursts (ms) */
    float calcium_decay;         /**< Ca2+ decay time constant (ms) */
} burst_params_t;

// ============================================================================
// UNIFIED TYPE PARAMETER UNION
// ============================================================================

/**
 * @brief Union of all type-specific parameters
 *
 * USAGE: Store in extended_params field of neuron
 * SIZE: Largest member determines size (~40 bytes)
 */
typedef union {
    visual_edge_params_t visual_edge;
    visual_orientation_params_t visual_orientation;
    auditory_frequency_params_t auditory_frequency;
    auditory_onset_params_t auditory_onset;
    motor_pattern_params_t motor_pattern;
    fast_spiking_params_t fast_spiking;
    burst_params_t burst;
} neuron_type_params_t;

// ============================================================================
// TYPE-SPECIFIC DYNAMICS FUNCTIONS
// ============================================================================

/**
 * @brief Compute type-specific input transformation
 *
 * @param type Neuron type
 * @param params Type-specific parameters
 * @param input Raw synaptic input
 * @param timestamp Current time (µs)
 *
 * @return Transformed input value
 *
 * DESIGN: Each type processes input differently:
 * - Visual edge: Apply Gabor filter
 * - Auditory frequency: Apply bandpass filter
 * - Motor pattern: Add intrinsic rhythm
 * - Fast-spiking: Fast dynamics
 */
float neuron_type_process_input(neuron_type_extended_t type,
                                const neuron_type_params_t* params,
                                float input,
                                uint64_t timestamp);

/**
 * @brief Get type-specific firing threshold adjustment
 *
 * @param type Neuron type
 * @param params Type-specific parameters
 * @param base_threshold Base threshold
 *
 * @return Adjusted threshold
 */
float neuron_type_get_threshold(neuron_type_extended_t type,
                                 const neuron_type_params_t* params,
                                 float base_threshold);

/**
 * @brief Get type-specific refractory period
 *
 * @param type Neuron type
 * @param params Type-specific parameters
 *
 * @return Refractory period (ms)
 */
float neuron_type_get_refractory_period(neuron_type_extended_t type,
                                         const neuron_type_params_t* params);

/**
 * @brief Check if neuron type is excitatory
 *
 * @param type Neuron type
 * @return true if excitatory, false if inhibitory
 *
 * DESIGN: Maps specialized types to E/I:
 * - Most specialized types are excitatory
 * - Interneurons (400-499) are inhibitory
 * - Motor Renshaw cells are inhibitory
 */
bool neuron_type_is_excitatory(neuron_type_extended_t type);

/**
 * @brief Get human-readable type name
 *
 * @param type Neuron type
 * @return String name (e.g., "Visual Edge Detector")
 */
const char* neuron_type_get_name(neuron_type_extended_t type);

// ============================================================================
// DEFAULT PARAMETERS
// ============================================================================

/**
 * @brief Get default parameters for a neuron type
 *
 * @param type Neuron type
 * @param out_params Output parameter structure
 *
 * @return NIMCP_SUCCESS on success
 */
nimcp_result_t neuron_type_get_default_params(neuron_type_extended_t type,
                                               neuron_type_params_t* out_params);

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * @brief Validate type-specific parameters
 *
 * @param type Neuron type
 * @param params Parameters to validate
 *
 * @return NIMCP_SUCCESS if valid
 */
nimcp_result_t neuron_type_validate_params(neuron_type_extended_t type,
                                            const neuron_type_params_t* params);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_NEURON_TYPES_H
