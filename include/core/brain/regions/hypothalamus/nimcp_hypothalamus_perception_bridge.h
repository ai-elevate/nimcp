/**
 * @file nimcp_hypothalamus_perception_bridge.h
 * @brief Hypothalamus-Perception Bridge for Sensory Modulation
 *
 * WHAT: Bidirectional integration between hypothalamus drives and sensory cortices
 * WHY:  Arousal and drives modulate sensory processing - hungry animals see food more
 * HOW:  Arousal → sensory gain; drive urgency → stimulus salience boost
 *
 * BIOLOGICAL BASIS:
 * The hypothalamus modulates sensory processing through multiple pathways:
 *
 * HYPOTHALAMUS → SENSORY CORTICES (via thalamus and neuromodulators):
 * - Arousal level → Global sensory gain (norepinephrine from LC)
 * - Drive urgency → Category-specific salience (dopamine from VTA)
 * - Threat detection → Fear-relevant stimuli bypass attention filter
 *
 * DRIVE-BIASED PERCEPTION:
 * - Hunger → Food-related stimuli more salient
 * - Thirst → Water/liquid stimuli more salient
 * - Safety threat → Threat-related stimuli prioritized
 * - Social drive → Face/voice stimuli enhanced
 * - Curiosity → Novel stimuli boosted
 *
 * SENSORY CORTICES → HYPOTHALAMUS:
 * - Detection of drive-relevant stimuli → Drive satisfaction anticipation
 * - Threat detection → Safety drive activation
 * - Food/water detection → Hunger/thirst anticipation
 *
 * @version Phase 17: Perception & Speech Modulation
 * @date 2026-01-04
 */

#ifndef NIMCP_HYPOTHALAMUS_PERCEPTION_BRIDGE_H
#define NIMCP_HYPOTHALAMUS_PERCEPTION_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "nimcp_hypothalamus_drives.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

/*=============================================================================
 * CONSTANTS
 *===========================================================================*/

#define HYPO_PERCEPTION_BRIDGE_MODULE_ID  0x1184
#define HYPO_PERCEPTION_MAX_CATEGORIES    16

/*=============================================================================
 * ENUMERATIONS
 *===========================================================================*/

/**
 * @brief Sensory modality types
 */
typedef enum {
    HYPO_SENSE_VISUAL = 0,        /**< Visual cortex (V1-V4, IT) */
    HYPO_SENSE_AUDITORY,          /**< Auditory cortex (A1, belt, parabelt) */
    HYPO_SENSE_SOMATOSENSORY,     /**< Somatosensory cortex (S1, S2) */
    HYPO_SENSE_OLFACTORY,         /**< Olfactory cortex (piriform) */
    HYPO_SENSE_GUSTATORY,         /**< Gustatory cortex (insula) */
    HYPO_SENSE_COUNT
} hypo_sense_modality_t;

/**
 * @brief Stimulus category types for drive-biased salience
 */
typedef enum {
    HYPO_STIM_FOOD = 0,           /**< Food-related stimuli */
    HYPO_STIM_WATER,              /**< Water/liquid stimuli */
    HYPO_STIM_THREAT,             /**< Threat-related stimuli */
    HYPO_STIM_SOCIAL,             /**< Social stimuli (faces, voices) */
    HYPO_STIM_NOVEL,              /**< Novel/unexpected stimuli */
    HYPO_STIM_THERMAL,            /**< Temperature-related stimuli */
    HYPO_STIM_PAIN,               /**< Nociceptive stimuli */
    HYPO_STIM_REWARD,             /**< Reward-associated stimuli */
    HYPO_STIM_NEUTRAL,            /**< Neutral/background stimuli */
    HYPO_STIM_COUNT
} hypo_stim_category_t;

/*=============================================================================
 * DATA STRUCTURES
 *===========================================================================*/

/**
 * @brief Sensory modulation output to cortices
 */
typedef struct {
    /* Global arousal modulation */
    float global_gain;            /**< Overall sensory gain [0.5, 2.0] */
    float arousal_level;          /**< Current arousal [0, 1] */

    /* Per-modality gain */
    float modality_gains[HYPO_SENSE_COUNT];

    /* Category-specific salience boosts */
    float category_salience[HYPO_STIM_COUNT];

    /* Attention override flags */
    bool threat_priority;         /**< Threat stimuli bypass attention gate */
    bool survival_mode;           /**< Critical drive = max sensory gain */

    /* Timestamp */
    uint64_t timestamp_us;
} hypo_perception_modulation_t;

/**
 * @brief Sensory detection feedback from cortices
 */
typedef struct {
    hypo_stim_category_t detected_category;  /**< What category was detected */
    hypo_sense_modality_t modality;          /**< Which modality detected it */
    float confidence;                        /**< Detection confidence [0, 1] */
    float intensity;                         /**< Stimulus intensity [0, 1] */
    bool is_threat;                          /**< Threat flag for fast path */
    uint64_t timestamp_us;
} hypo_sensory_detection_t;

/**
 * @brief Bridge configuration
 */
typedef struct {
    /* Arousal → gain mapping */
    float arousal_gain_min;       /**< Gain at zero arousal (default 0.7) */
    float arousal_gain_max;       /**< Gain at max arousal (default 1.5) */

    /* Drive → salience mapping */
    float drive_salience_weight;  /**< How much drives boost salience (default 0.8) */
    float threat_priority_threshold; /**< Safety urgency for threat priority (default 0.5) */

    /* Modality-specific weights */
    float visual_weight;          /**< Visual modality sensitivity */
    float auditory_weight;        /**< Auditory modality sensitivity */

    /* Anticipation */
    bool enable_anticipation;     /**< Enable drive anticipation from detection */
    float anticipation_decay;     /**< How fast anticipation decays */
} hypo_perception_config_t;

typedef struct hypo_perception_bridge hypo_perception_bridge_t;

/*=============================================================================
 * LIFECYCLE
 *==============================================================================*/

/**
 * @brief Create perception bridge
 *
 * @param drives Drive system handle
 * @param config Configuration (NULL for defaults)
 * @return Bridge handle or NULL on failure
 */
hypo_perception_bridge_t* hypo_perception_bridge_create(
    hypo_drive_system_handle_t* drives,
    const hypo_perception_config_t* config);

/**
 * @brief Destroy perception bridge
 */
void hypo_perception_bridge_destroy(hypo_perception_bridge_t* bridge);

/**
 * @brief Get default configuration
 */
void hypo_perception_bridge_default_config(hypo_perception_config_t* config);

/*=============================================================================
 * DRIVE → PERCEPTION MODULATION
 *===========================================================================*/

/**
 * @brief Compute perception modulation from current drive state
 *
 * Maps arousal and drive urgencies to sensory modulation parameters:
 * - Global gain from arousal level
 * - Category salience from corresponding drive urgencies
 * - Threat priority from safety drive
 *
 * @param bridge Bridge handle
 * @return 0 on success, -1 on error
 */
int hypo_perception_bridge_compute_modulation(hypo_perception_bridge_t* bridge);

/**
 * @brief Get current perception modulation
 *
 * @param bridge Bridge handle
 * @param modulation Output modulation parameters
 * @return 0 on success, -1 on error
 */
int hypo_perception_bridge_get_modulation(
    const hypo_perception_bridge_t* bridge,
    hypo_perception_modulation_t* modulation);

/**
 * @brief Set arousal level (from brainstem/LC)
 *
 * @param bridge Bridge handle
 * @param arousal Arousal level [0, 1]
 * @return 0 on success, -1 on error
 */
int hypo_perception_bridge_set_arousal(
    hypo_perception_bridge_t* bridge,
    float arousal);

/*=============================================================================
 * PERCEPTION → DRIVE FEEDBACK
 *===========================================================================*/

/**
 * @brief Process sensory detection
 *
 * When sensory cortices detect drive-relevant stimuli, this informs
 * the drive system for anticipation effects.
 *
 * @param bridge Bridge handle
 * @param detection Detection event
 * @return 0 on success, -1 on error
 */
int hypo_perception_bridge_process_detection(
    hypo_perception_bridge_t* bridge,
    const hypo_sensory_detection_t* detection);

/**
 * @brief Get anticipation level for a drive
 *
 * Returns how much the drive is being anticipated based on
 * recent sensory detections (seeing food increases hunger anticipation).
 *
 * @param bridge Bridge handle
 * @param drive_type Which drive
 * @return Anticipation level [0, 1]
 */
float hypo_perception_bridge_get_anticipation(
    const hypo_perception_bridge_t* bridge,
    hypo_drive_type_t drive_type);

/*=============================================================================
 * BIO-ASYNC INTEGRATION
 *===========================================================================*/

/**
 * @brief Register with bio-async router
 *
 * @param bridge Bridge handle
 * @param use_kg_wiring Use knowledge graph wiring
 * @return true on success
 */
bool hypo_perception_bridge_register_bio(
    hypo_perception_bridge_t* bridge,
    bool use_kg_wiring);

/**
 * @brief Unregister from bio-async router
 */
void hypo_perception_bridge_unregister_bio(hypo_perception_bridge_t* bridge);

/**
 * @brief Broadcast perception modulation
 *
 * @param bridge Bridge handle
 * @return NIMCP_SUCCESS on success
 */
nimcp_error_t hypo_perception_bridge_broadcast_modulation(
    hypo_perception_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_HYPOTHALAMUS_PERCEPTION_BRIDGE_H */
