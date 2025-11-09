/**
 * @file nimcp_emotional_tagging.c
 * @brief Implementation of emotional tagging system
 *
 * WHAT: Tags cognitive representations with emotional states
 * WHY:  Emotions enhance memory and guide attention
 * HOW:  Russell's circumplex model (valence × arousal)
 *
 * CODING STANDARDS:
 * - All functions < 50 lines
 * - Guard clauses (no nested ifs)
 * - Single Responsibility Principle
 * - WHAT-WHY-HOW documentation
 */

#include "cognitive/nimcp_emotional_tagging.h"
#include <math.h>
#include <string.h>

//=============================================================================
// Helper Functions (Internal, <50 lines each)
//=============================================================================

/**
 * @brief Clamp float to range [min, max]
 *
 * WHAT: Restrict value to valid range
 * WHY:  Prevent invalid emotional coordinates
 * HOW:  Standard clamping algorithm
 */
static inline float clamp(float value, float min, float max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Compute absolute value
 *
 * WHAT: Get magnitude without sign
 * WHY:  Valence intensity (positive or negative both matter)
 * HOW:  Standard fabs()
 */
static inline float abs_float(float value)
{
    return fabsf(value);
}

/**
 * @brief Check if emotion is high arousal
 *
 * WHAT: Test if arousal exceeds threshold
 * WHY:  Classify emotions into arousal categories
 * HOW:  Compare to 0.5 threshold
 */
static inline bool is_high_arousal(float arousal)
{
    return arousal > 0.5f;
}

/**
 * @brief Check if emotion is positive valence
 *
 * WHAT: Test if valence is positive
 * WHY:  Classify emotions into valence categories
 * HOW:  Compare to threshold
 */
static inline bool is_positive_valence(float valence)
{
    return valence > 0.2f;
}

/**
 * @brief Check if emotion is negative valence
 *
 * WHAT: Test if valence is negative
 * WHY:  Classify emotions into valence categories
 * HOW:  Compare to threshold
 */
static inline bool is_negative_valence(float valence)
{
    return valence < -0.2f;
}

/**
 * @brief Classify high-arousal positive emotions
 *
 * WHAT: Determine if JOY or EXCITEMENT
 * WHY:  Distinguish between high-arousal positive states
 * HOW:  Threshold on valence strength
 */
static emotion_category_t classify_high_arousal_positive(float valence, float arousal)
{
    // Guard: Very high valence → JOY
    if (valence > 0.5f && arousal > 0.5f) {
        return EMOTION_JOY;
    }

    // High arousal, moderate positive → EXCITEMENT
    return EMOTION_EXCITEMENT;
}

/**
 * @brief Classify high-arousal negative emotions
 *
 * WHAT: Determine if FEAR, ANGER, or ANXIETY
 * WHY:  Distinguish between high-arousal negative states
 * HOW:  Threshold on valence and arousal strength
 */
static emotion_category_t classify_high_arousal_negative(float valence, float arousal)
{
    // Guard: Very negative + very high arousal → FEAR
    if (valence < -0.5f && arousal > 0.7f) {
        return EMOTION_FEAR;
    }

    // Guard: Moderately negative + high arousal → ANGER
    if (valence < -0.4f) {
        return EMOTION_ANGER;
    }

    // Default: ANXIETY (negative + aroused but not extreme)
    return EMOTION_ANXIETY;
}

/**
 * @brief Classify low-arousal emotions
 *
 * WHAT: Determine if CALM, SADNESS, BOREDOM, or NEUTRAL
 * WHY:  Distinguish between low-arousal states
 * HOW:  Threshold on valence
 */
static emotion_category_t classify_low_arousal(float valence, float arousal)
{
    // Guard: Positive + low arousal → CALM
    if (valence > 0.2f) {
        return EMOTION_CALM;
    }

    // Guard: Negative + very low arousal → BOREDOM
    if (valence < -0.1f && arousal < 0.2f) {
        return EMOTION_BOREDOM;
    }

    // Guard: Moderately negative + low arousal → SADNESS
    if (valence < -0.3f) {
        return EMOTION_SADNESS;
    }

    // Default: NEUTRAL (near origin)
    return EMOTION_NEUTRAL;
}

//=============================================================================
// Public API Implementation
//=============================================================================

/**
 * @brief Create emotional tag from valence and arousal
 *
 * WHAT: Initialize emotional tag structure
 * WHY:  Primary constructor for emotional states
 * HOW:  Clamp values → classify → compute intensity
 */
emotional_tag_t emotional_tag_create(
    float valence,
    float arousal,
    uint64_t timestamp_ms)
{
    emotional_tag_t tag;

    // Clamp to valid ranges
    tag.valence = clamp(valence, -1.0f, 1.0f);
    tag.arousal = clamp(arousal, 0.0f, 1.0f);
    tag.timestamp_ms = timestamp_ms;

    // Classify emotion category
    tag.category = emotional_tag_classify(&tag);

    // Compute intensity
    tag.intensity = emotional_tag_intensity(&tag);

    return tag;
}

/**
 * @brief Create neutral emotional tag
 *
 * WHAT: Initialize with zero emotion
 * WHY:  Convenience for neutral states
 * HOW:  Set all fields to neutral/zero
 */
emotional_tag_t emotional_tag_neutral(void)
{
    emotional_tag_t tag = {
        .valence = 0.0f,
        .arousal = 0.0f,
        .timestamp_ms = 0,
        .category = EMOTION_NEUTRAL,
        .intensity = 0.0f
    };
    return tag;
}

/**
 * @brief Classify emotion into discrete category
 *
 * WHAT: Map 2D coordinates to emotion category
 * WHY:  Provide human-readable emotional state
 * HOW:  Hierarchical classification with guard clauses
 *
 * COMPLEXITY: O(1) - ~10 comparisons maximum
 */
emotion_category_t emotional_tag_classify(const emotional_tag_t* tag)
{
    // Guard: NULL tag
    if (!tag) {
        return EMOTION_NEUTRAL;
    }

    // Extract values
    float valence = tag->valence;
    float arousal = tag->arousal;

    // High arousal emotions
    if (is_high_arousal(arousal)) {
        if (is_positive_valence(valence)) {
            return classify_high_arousal_positive(valence, arousal);
        }
        if (is_negative_valence(valence)) {
            return classify_high_arousal_negative(valence, arousal);
        }
    }

    // Low arousal emotions
    return classify_low_arousal(valence, arousal);
}

/**
 * @brief Compute emotional intensity
 *
 * WHAT: Vector magnitude in valence-arousal space
 * WHY:  Single scalar for emotional strength
 * HOW:  Euclidean distance from origin, normalized
 *
 * FORMULA: sqrt(valence² + arousal²) / sqrt(2)
 * RANGE: [0.0, 1.0]
 */
float emotional_tag_intensity(const emotional_tag_t* tag)
{
    // Guard: NULL tag
    if (!tag) {
        return 0.0f;
    }

    // Compute magnitude
    float val_sq = tag->valence * tag->valence;
    float aro_sq = tag->arousal * tag->arousal;
    float magnitude = sqrtf(val_sq + aro_sq);

    // Normalize to [0, 1] (max possible magnitude is sqrt(2))
    return magnitude / 1.41421356f;  // sqrt(2)
}

/**
 * @brief Get emotion category name
 *
 * WHAT: Convert enum to string
 * WHY:  Debugging and logging
 * HOW:  Static string array lookup
 */
const char* emotional_category_name(emotion_category_t category)
{
    static const char* names[] = {
        "NEUTRAL",
        "JOY",
        "EXCITEMENT",
        "CALM",
        "FEAR",
        "ANGER",
        "SADNESS",
        "ANXIETY",
        "BOREDOM"
    };

    // Guard: Invalid category
    if (category < 0 || category >= 9) {
        return "UNKNOWN";
    }

    return names[category];
}

/**
 * @brief Compute salience boost from emotion
 *
 * WHAT: Multiplier for base salience
 * WHY:  Emotional events grab attention
 * HOW:  Arousal and valence intensity both contribute
 *
 * FORMULA: 1.0 + (arousal × 0.5) + (|valence| × 0.3)
 * RANGE: [1.0, 1.8]
 */
float emotional_compute_salience_boost(const emotional_tag_t* tag)
{
    // Guard: NULL tag → no boost
    if (!tag) {
        return 1.0f;
    }

    // Arousal contribution (up to +0.5)
    float arousal_boost = tag->arousal * EMOTIONAL_AROUSAL_SALIENCE_FACTOR;

    // Valence intensity contribution (up to +0.3)
    float valence_boost = abs_float(tag->valence) * EMOTIONAL_VALENCE_SALIENCE_FACTOR;

    // Total boost (baseline 1.0 + contributions)
    return 1.0f + arousal_boost + valence_boost;
}

/**
 * @brief Apply emotional boost to salience
 *
 * WHAT: Compute final salience with emotion
 * WHY:  Convenience function
 * HOW:  Multiply base × boost
 */
float emotional_apply_salience_boost(
    float base_salience,
    const emotional_tag_t* tag)
{
    // Guard: Invalid base salience
    if (base_salience < 0.0f) {
        return 0.0f;
    }

    float boost = emotional_compute_salience_boost(tag);
    return base_salience * boost;
}

/**
 * @brief Detect emotion from cognitive state
 *
 * WHAT: Infer emotional tag from processing outputs
 * WHY:  Automatic emotion tagging during inference
 * HOW:  Map cognitive signals to valence-arousal space
 *
 * MAPPINGS:
 * - confidence → valence
 * - uncertainty → arousal
 * - novelty → positive valence + arousal
 * - ethical violation → negative valence
 */
emotional_tag_t emotional_tag_from_cognitive_state(
    float confidence,
    float uncertainty,
    float novelty,
    bool ethical_approved,
    uint64_t timestamp_ms)
{
    // Initialize with neutral
    float valence = 0.0f;
    float arousal = 0.0f;

    // Map confidence to valence: high confidence = positive
    // confidence ∈ [0,1] → valence ∈ [-1,+1]
    valence = (confidence - 0.5f) * 2.0f;

    // Map uncertainty to arousal: high uncertainty = high arousal
    arousal = clamp(uncertainty * 1.2f, 0.0f, 1.0f);

    // Novelty adds positive valence and arousal (curiosity)
    if (novelty > 0.5f) {
        valence += (novelty - 0.5f) * 0.6f;  // Boost positive valence
        arousal += (novelty - 0.5f) * 0.4f;   // Boost arousal
    }

    // Ethical violation → strong negative valence + high arousal
    if (!ethical_approved) {
        valence = -0.8f;  // Strong negative
        arousal = clamp(arousal + 0.3f, 0.0f, 1.0f);  // Increase arousal
    }

    // Create tag with computed values
    return emotional_tag_create(valence, arousal, timestamp_ms);
}

/**
 * @brief Validate emotional tag
 *
 * WHAT: Check if values are in valid ranges
 * WHY:  Prevent invalid emotional states
 * HOW:  Range checks
 */
bool emotional_tag_is_valid(const emotional_tag_t* tag)
{
    // Guard: NULL tag
    if (!tag) {
        return false;
    }

    // Check ranges
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

/**
 * @brief Clamp emotional tag to valid ranges
 *
 * WHAT: Force values into valid bounds
 * WHY:  Sanitize potentially invalid states
 * HOW:  Clamp each field
 */
void emotional_tag_clamp(emotional_tag_t* tag)
{
    // Guard: NULL tag
    if (!tag) {
        return;
    }

    tag->valence = clamp(tag->valence, -1.0f, 1.0f);
    tag->arousal = clamp(tag->arousal, 0.0f, 1.0f);
    tag->intensity = clamp(tag->intensity, 0.0f, 1.0f);

    // Recompute category and intensity after clamping
    tag->category = emotional_tag_classify(tag);
    tag->intensity = emotional_tag_intensity(tag);
}
