/**
 * @file nimcp_emotional_tagging.c
 * @brief Emotional tagging system implementation - Russell's circumplex model
 *
 * WHAT: Implements emotional tagging for cognitive representations
 * WHY:  Emotions enhance memory consolidation and guide attention
 * HOW:  Maps valence-arousal coordinates to emotion categories, computes salience
 *
 * BIOLOGICAL BASIS:
 * - Amygdala emotional tagging modulates hippocampal encoding
 * - High arousal events get attentional priority
 * - Emotional context aids memory retrieval
 *
 * @author NIMCP Development Team - Phase 10.3
 * @date 2025-11-20
 * @version 2.7.0 Phase 10.3
 */

#include "cognitive/nimcp_emotional_tagging.h"
#include <math.h>
#include <string.h>

//=============================================================================
// Helper Macros
//=============================================================================

#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define CLAMP_01(x) CLAMP(x, 0.0f, 1.0f)
#define CLAMP_VALENCE(x) CLAMP(x, -1.0f, 1.0f)

//=============================================================================
// Lifecycle Functions
//=============================================================================

emotional_tag_t emotional_tag_create(
    float valence,
    float arousal,
    uint64_t timestamp_ms
) {
    emotional_tag_t tag;

    /* WHAT: Clamp inputs to valid ranges
     * WHY:  Prevent invalid emotional states
     * HOW:  Use CLAMP macros for valence and arousal */
    tag.valence = CLAMP_VALENCE(valence);
    tag.arousal = CLAMP_01(arousal);
    tag.timestamp_ms = timestamp_ms;

    /* WHAT: Classify emotion category from valence-arousal
     * WHY:  Provide human-readable emotional state
     * HOW:  Call classification function */
    tag.category = emotional_tag_classify(&tag);

    /* WHAT: Compute overall intensity
     * WHY:  Single scalar for emotional strength
     * HOW:  Vector magnitude in 2D space */
    tag.intensity = emotional_tag_intensity(&tag);

    return tag;
}

emotional_tag_t emotional_tag_neutral(void) {
    emotional_tag_t tag;
    tag.valence = 0.0f;
    tag.arousal = 0.0f;
    tag.timestamp_ms = 0;
    tag.category = EMOTION_NEUTRAL;
    tag.intensity = 0.0f;
    return tag;
}

//=============================================================================
// Classification & Analysis
//=============================================================================

emotion_category_t emotional_tag_classify(const emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return EMOTION_NEUTRAL;
    }

    float v = tag->valence;
    float a = tag->arousal;

    /* WHAT: Classify based on Russell's circumplex model
     * WHY:  Map continuous 2D space to discrete categories
     * HOW:  Threshold-based quadrant classification */

    /* Low arousal region */
    if (a < 0.3f) {
        if (v > 0.2f) return EMOTION_CALM;
        if (v < 0.0f && a < 0.2f) return EMOTION_BOREDOM;
        return EMOTION_NEUTRAL;
    }

    /* High arousal + positive valence */
    if (v > 0.5f && a > 0.5f) {
        return EMOTION_JOY;
    }
    if (v > 0.3f && a > 0.7f) {
        return EMOTION_EXCITEMENT;
    }

    /* High arousal + negative valence */
    if (v < -0.3f && a > 0.6f) {
        if (v < -0.4f && a > 0.6f) {
            return EMOTION_ANGER;
        }
        return EMOTION_FEAR;
    }
    if (v < -0.2f && a > 0.5f) {
        return EMOTION_ANXIETY;
    }

    /* Low-moderate arousal + negative valence */
    if (v < -0.3f && a < 0.4f) {
        return EMOTION_SADNESS;
    }

    return EMOTION_NEUTRAL;
}

float emotional_tag_intensity(const emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return 0.0f;
    }

    /* WHAT: Compute vector magnitude in valence-arousal space
     * WHY:  Combined measure of emotional strength
     * HOW:  Euclidean distance, normalized by sqrt(2) */
    float intensity = sqrtf(tag->valence * tag->valence +
                           tag->arousal * tag->arousal) / sqrtf(2.0f);

    return CLAMP_01(intensity);
}

const char* emotional_category_name(emotion_category_t category) {
    /* WHAT: Static string lookup for emotion names
     * WHY:  Human-readable debugging output
     * HOW:  Switch statement on enum */
    switch (category) {
        case EMOTION_NEUTRAL:    return "NEUTRAL";
        case EMOTION_JOY:        return "JOY";
        case EMOTION_EXCITEMENT: return "EXCITEMENT";
        case EMOTION_CALM:       return "CALM";
        case EMOTION_FEAR:       return "FEAR";
        case EMOTION_ANGER:      return "ANGER";
        case EMOTION_SADNESS:    return "SADNESS";
        case EMOTION_ANXIETY:    return "ANXIETY";
        case EMOTION_BOREDOM:    return "BOREDOM";
        default:                 return "UNKNOWN";
    }
}

//=============================================================================
// Salience Modulation
//=============================================================================

float emotional_compute_salience_boost(const emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return 1.0f;
    }

    /* WHAT: Compute salience multiplier from emotion
     * WHY:  Emotional events get attentional priority
     * HOW:  Arousal and valence intensity both boost salience */

    float arousal_boost = tag->arousal * EMOTIONAL_AROUSAL_SALIENCE_FACTOR;
    float valence_boost = fabsf(tag->valence) * EMOTIONAL_VALENCE_SALIENCE_FACTOR;

    /* Base 1.0 + emotional boost */
    return 1.0f + arousal_boost + valence_boost;
}

float emotional_apply_salience_boost(
    float base_salience,
    const emotional_tag_t* tag
) {
    /* Guard clause: Validate input */
    if (!tag || base_salience < 0.0f) {
        return base_salience;
    }

    float boost = emotional_compute_salience_boost(tag);
    return base_salience * boost;
}

//=============================================================================
// Emotion Detection from Cognitive State
//=============================================================================

emotional_tag_t emotional_tag_from_cognitive_state(
    float confidence,
    float uncertainty,
    float novelty,
    bool ethical_approved,
    uint64_t timestamp_ms
) {
    /* WHAT: Map cognitive assessment to valence-arousal
     * WHY:  Automatic emotion detection during processing
     * HOW:  Heuristic mappings from cognitive signals */

    float valence = 0.0f;
    float arousal = 0.0f;

    /* WHAT: Confidence → positive valence
     * WHY:  High confidence feels good */
    valence += (confidence - 0.5f) * 0.8f;  // Map [0,1] → [-0.4, +0.4]

    /* WHAT: Novelty → positive valence + arousal
     * WHY:  Curiosity is pleasant and activating */
    valence += novelty * 0.3f;
    arousal += novelty * 0.4f;

    /* WHAT: Uncertainty → arousal
     * WHY:  Uncertainty demands attention */
    arousal += uncertainty * 0.5f;

    /* WHAT: Ethical violation → strong negative valence
     * WHY:  Ethical concerns are emotionally aversive */
    if (!ethical_approved) {
        valence -= 0.6f;
        arousal += 0.3f;
    }

    return emotional_tag_create(valence, arousal, timestamp_ms);
}

//=============================================================================
// Validation
//=============================================================================

bool emotional_tag_is_valid(const emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return false;
    }

    /* WHAT: Check ranges for valence, arousal, intensity
     * WHY:  Detect corrupted or invalid emotional states
     * HOW:  Range checks on each component */

    if (tag->valence < -1.0f || tag->valence > 1.0f) {
        return false;
    }
    if (tag->arousal < 0.0f || tag->arousal > 1.0f) {
        return false;
    }
    if (tag->intensity < 0.0f || tag->intensity > 1.0f) {
        return false;
    }

    return true;
}

void emotional_tag_clamp(emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return;
    }

    /* WHAT: Force emotional tag into valid ranges
     * WHY:  Sanitize potentially invalid states
     * HOW:  Clamp each component */

    tag->valence = CLAMP_VALENCE(tag->valence);
    tag->arousal = CLAMP_01(tag->arousal);
    tag->intensity = CLAMP_01(tag->intensity);
}

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

float emotional_get_valence(const emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return 0.0f;
    }

    return tag->valence;
}

float emotional_get_arousal(const emotional_tag_t* tag) {
    /* Guard clause: Validate input */
    if (!tag) {
        return 0.0f;
    }

    return tag->arousal;
}

void emotional_modulate_arousal(emotional_tag_t* tag, float arousal_delta) {
    /* Guard clause: Validate input */
    if (!tag) {
        return;
    }

    /* WHAT: Modify arousal level
     * WHY:  Allow salient events to increase emotional arousal
     * HOW:  Add delta and clamp to valid range */

    tag->arousal += arousal_delta;
    tag->arousal = CLAMP_01(tag->arousal);

    /* WHAT: Recompute derived fields
     * WHY:  Keep category and intensity consistent
     * HOW:  Reclassify and recompute intensity */
    tag->category = emotional_tag_classify(tag);
    tag->intensity = emotional_tag_intensity(tag);
}
