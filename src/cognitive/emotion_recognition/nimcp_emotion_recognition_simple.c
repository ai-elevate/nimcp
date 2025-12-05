// ============================================================================
// nimcp_emotion_recognition_simple.c - Simple Text Emotion Recognition
// ============================================================================
/**
 * @file nimcp_emotion_recognition_simple.c
 * @brief Lightweight text-based emotion recognition for integration
 *
 * WHAT: Simple keyword-based emotion detection from text
 * WHY:  Provide basic emotion recognition without full multimodal system
 * HOW:  Keyword matching + sentiment analysis
 *
 * NOTE: This is a lightweight implementation to enable empathetic response
 *       integration. Full multimodal emotion recognition (nimcp_emotion_recognition.h)
 *       can replace this when implemented.
 */

#include <string.h>
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>

// ============================================================================
// Emotion Keywords
// ============================================================================

static const char* ANGER_KEYWORDS[] = {
    "angry", "mad", "furious", "annoyed", "frustrated", "irritated",
    "hate", "stupid", "damn", "ridiculous", "terrible", NULL
};

static const char* FEAR_KEYWORDS[] = {
    "afraid", "scared", "worried", "anxious", "nervous", "terrified",
    "panic", "frightened", "concerned", NULL
};

static const char* SADNESS_KEYWORDS[] = {
    "sad", "unhappy", "depressed", "miserable", "hopeless", "crying",
    "tears", "lonely", "hurt", "pain", NULL
};

static const char* HAPPINESS_KEYWORDS[] = {
    "happy", "glad", "excited", "great", "wonderful", "amazing",
    "awesome", "fantastic", "love", "joy", NULL
};

static const char* CONFUSION_KEYWORDS[] = {
    "confused", "don't understand", "what", "how", "unclear",
    "lost", "puzzled", NULL
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Convert text to lowercase for matching
 */
static void to_lowercase(char* dest, const char* src, size_t max_len)
{
    size_t i;
    for (i = 0; i < max_len - 1 && src[i] != '\0'; i++) {
        dest[i] = tolower((unsigned char)src[i]);
    }
    dest[i] = '\0';
}

/**
 * @brief Count keywords in text
 */
static int count_keywords(const char* text, const char* keywords[])
{
    char lowercase_text[512];
    to_lowercase(lowercase_text, text, sizeof(lowercase_text));

    int count = 0;
    for (int i = 0; keywords[i] != NULL; i++) {
        if (strstr(lowercase_text, keywords[i]) != NULL) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Simple text-based emotion recognition
 *
 * WHAT: Detect emotion from text using keyword matching
 * WHY:  Enable empathetic response without full emotion recognition system
 * HOW:  Count emotion keywords, select dominant emotion
 *
 * @param text Input text to analyze
 * @param emotion_name Output: Detected emotion name
 * @param emotion_name_len Size of emotion_name buffer
 * @param confidence Output: Confidence [0,1]
 * @param valence Output: Valence [-1,1] (negative to positive)
 * @param arousal Output: Arousal [0,1] (calm to excited)
 * @return true if emotion detected, false otherwise
 *
 * COMPLEXITY: O(n) where n = text length
 */
bool emotion_recognize_text_simple(
    const char* text,
    char* emotion_name,
    size_t emotion_name_len,
    float* confidence,
    float* valence,
    float* arousal)
{
    // Guard: NULL checks
    if (!text || !emotion_name || !confidence || !valence || !arousal) {
        return false;
    }

    // Initialize outputs
    emotion_name[0] = '\0';
    *confidence = 0.0f;
    *valence = 0.0f;
    *arousal = 0.0f;

    // Guard: Empty text
    if (text[0] == '\0') {
        strncpy(emotion_name, "neutral", emotion_name_len - 1);
        *confidence = 0.5f;
        *valence = 0.0f;
        *arousal = 0.0f;
        return true;
    }

    // Count keywords per emotion
    int anger_count = count_keywords(text, ANGER_KEYWORDS);
    int fear_count = count_keywords(text, FEAR_KEYWORDS);
    int sadness_count = count_keywords(text, SADNESS_KEYWORDS);
    int happiness_count = count_keywords(text, HAPPINESS_KEYWORDS);
    int confusion_count = count_keywords(text, CONFUSION_KEYWORDS);

    // Select dominant emotion
    int max_count = 0;
    const char* detected_emotion = "neutral";
    float detected_valence = 0.0f;
    float detected_arousal = 0.3f;

    if (anger_count > max_count) {
        max_count = anger_count;
        detected_emotion = "anger";
        detected_valence = -0.7f;
        detected_arousal = 0.8f;
    }
    if (fear_count > max_count) {
        max_count = fear_count;
        detected_emotion = "fear";
        detected_valence = -0.6f;
        detected_arousal = 0.9f;
    }
    if (sadness_count > max_count) {
        max_count = sadness_count;
        detected_emotion = "sadness";
        detected_valence = -0.6f;
        detected_arousal = 0.2f;
    }
    if (happiness_count > max_count) {
        max_count = happiness_count;
        detected_emotion = "happiness";
        detected_valence = 0.8f;
        detected_arousal = 0.6f;
    }
    if (confusion_count > max_count) {
        max_count = confusion_count;
        detected_emotion = "confusion";
        detected_valence = -0.2f;
        detected_arousal = 0.5f;
    }

    // Calculate confidence based on keyword density
    size_t text_len = strlen(text);
    float keyword_density = (max_count > 0) ?
        ((float)max_count / (text_len / 20.0f)) : 0.0f;

    // Confidence: 0.5 base + up to 0.4 based on keyword density
    *confidence = (max_count > 0) ? (0.5f + (keyword_density * 0.4f)) : 0.3f;
    if (*confidence > 1.0f) *confidence = 1.0f;

    // Set outputs
    strncpy(emotion_name, detected_emotion, emotion_name_len - 1);
    emotion_name[emotion_name_len - 1] = '\0';
    *valence = detected_valence;
    *arousal = detected_arousal;

    return true;
}
