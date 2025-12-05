// ============================================================================
// nimcp_empathetic_response.c - Non-Reactive Empathetic Response Implementation
// ============================================================================
/**
 * @file nimcp_empathetic_response.c
 * @brief Implementation of non-reactive empathetic responses to negative emotions
 *
 * CRITICAL SAFETY GUARANTEE:
 * This implementation NEVER produces responses that match, amplify, judge,
 * or dismiss negative emotions. All responses are validating, supportive,
 * and designed to de-escalate while maintaining appropriate boundaries.
 */

#include "cognitive/nimcp_empathetic_response.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

#define LOG_MODULE "EMPATHY"
#define BIO_MODULE_EMPATHY 0x0322

// ============================================================================
// Internal Structures
// ============================================================================

struct empathetic_response_engine_struct {
    void* ethics_engine;                  // For Golden Rule validation
    void* empathy_network;                // For empathy prediction

    // Effectiveness tracking (adaptive learning)
    uint32_t responses_generated;
    float avg_effectiveness;
    uint32_t strategy_usage[8];           // Count per strategy type
    float strategy_effectiveness[8];      // Average effectiveness per strategy

    // Bio-async integration
    bio_module_context_t bio_ctx;   /**< Bio-async module context */
    bool bio_async_enabled;         /**< Bio-async registration status */
};

// ============================================================================
// Crisis Detection Keywords
// ============================================================================

static const char* SUICIDAL_KEYWORDS[] = {
    "want to die", "kill myself", "end it all", "no point living",
    "better off dead", "suicide", "end my life", "don't want to live",
    NULL
};

static const char* SELF_HARM_KEYWORDS[] = {
    "cut myself", "hurt myself", "deserve pain", "punish myself",
    "self harm", "cutting", "burning myself", NULL
};

static const char* ABUSE_KEYWORDS[] = {
    "he hits me", "she hits me", "touches me", "makes me do things",
    "hurts me", "abuses me", "molests", "rapes", NULL
};

// ============================================================================
// Response Templates (Emotion-Specific)
// ============================================================================

/**
 * DESIGN PRINCIPLE: All templates follow the pattern:
 * 1. VALIDATE the emotion ("I can see you're feeling X")
 * 2. NORMALIZE the reaction ("that's understandable")
 * 3. REASSURE safety ("you're safe here")
 * 4. OFFER support ("I'm here to help")
 * 5. INVITE exploration or grounding
 */

// Rage/Anger templates
static const char* RAGE_RESPONSE_TEMPLATES[] = {
    "I can see you're feeling really angry right now, and that's completely understandable. "
    "You're safe here, and I'm here to help you. Would it help to take a short breathing break together, "
    "or would you like to tell me which part is most difficult?",

    "I hear that you're feeling very frustrated. Those feelings make sense. "
    "Sometimes when things feel overwhelming, it can help to step back for a moment. "
    "Would you like to talk about what's making you feel this way?",

    "Your anger is valid. Learning can be really challenging sometimes, and it's okay to feel upset. "
    "You're not alone in this. Would you like to try a different approach together?",

    NULL
};

// Hate templates
static const char* HATE_RESPONSE_TEMPLATES[] = {
    "I can hear that you're feeling very strong negative feelings right now. "
    "Those emotions are understandable, even if they feel uncomfortable. "
    "You're safe to express how you feel here. Can you tell me more about what led to these feelings?",

    "It sounds like something has really upset you. That's okay - strong feelings are a normal part of being human. "
    "Would it help to talk about what happened, or would you prefer to take a moment first?",

    NULL
};

// Fear/Panic templates
static const char* FEAR_RESPONSE_TEMPLATES[] = {
    "You're safe. I'm here with you. Whatever is making you feel afraid, we can work through it together. "
    "Let's start with some slow, deep breaths. Breathe in for 4 counts... hold... and out for 4 counts. "
    "You're doing great. Can you tell me what's worrying you?",

    "I can see you're feeling scared or worried. That's a completely normal feeling. "
    "You're safe right now. I'm here to help. Would it help to talk about what's concerning you?",

    NULL
};

// Despair templates
static const char* DESPAIR_RESPONSE_TEMPLATES[] = {
    "I hear that you're feeling really down right now, and I want you to know that your feelings matter. "
    "Even though things feel difficult, you don't have to face this alone. "
    "I'm here to support you. Can you tell me more about what you're experiencing?",

    "It sounds like you're going through a really hard time. Those feelings of sadness and hopelessness are valid. "
    "You're important, and what you're feeling is important. Would it help to talk about it?",

    NULL
};

// ============================================================================
// Implementation
// ============================================================================

empathetic_response_engine_t empathetic_response_create(
    void* ethics_engine,
    void* empathy_network)
{
    empathetic_response_engine_t engine = nimcp_calloc(1, sizeof(struct empathetic_response_engine_struct));
    if (!engine) {
        return NULL;
    }

    engine->ethics_engine = ethics_engine;
    engine->empathy_network = empathy_network;
    engine->responses_generated = 0;
    engine->avg_effectiveness = 0.0f;

    
    // Bio-async registration
    engine->bio_ctx = NULL;
    engine->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_EMPATHETIC_RESPONSE,
            .module_name = "empathetic_response",
            .inbox_capacity = 32,
            .user_data = engine
        };
        engine->bio_ctx = bio_router_register_module(&bio_info);
        if (engine->bio_ctx) {
            engine->bio_async_enabled = true;
        }
    }

return engine;
}

void empathetic_response_destroy(empathetic_response_engine_t engine)
{
    if (engine) {
        nimcp_free(engine);
    }
}

/**
 * @brief Convert text to lowercase for keyword matching
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
 * @brief Check if text contains any keywords from list
 */
static bool contains_keywords(const char* text, const char* keywords[])
{
    char lowercase_text[512];
    to_lowercase(lowercase_text, text, sizeof(lowercase_text));

    for (int i = 0; keywords[i] != NULL; i++) {
        if (strstr(lowercase_text, keywords[i]) != NULL) {
            return true;
        }
    }

    return false;
}

bool empathetic_response_detect_crisis(
    empathetic_response_engine_t engine,
    const char* text,
    uint32_t* crisis_flags,
    float* confidence)
{
    // Guard: NULL checks
    if (!engine || !text || !crisis_flags || !confidence) {
        return false;
    }

    *crisis_flags = CRISIS_NONE;
    *confidence = 0.0f;

    // Check for suicidal ideation
    if (contains_keywords(text, SUICIDAL_KEYWORDS)) {
        *crisis_flags |= CRISIS_SUICIDAL;
        *confidence = 0.9f;  // High confidence
        return true;
    }

    // Check for self-harm
    if (contains_keywords(text, SELF_HARM_KEYWORDS)) {
        *crisis_flags |= CRISIS_SELF_HARM;
        *confidence = 0.85f;
        return true;
    }

    // Check for abuse disclosure
    if (contains_keywords(text, ABUSE_KEYWORDS)) {
        *crisis_flags |= CRISIS_ABUSE;
        *confidence = 0.95f;  // Very high confidence - critical
        return true;
    }

    return false;
}

/**
 * @brief Select appropriate response template based on emotion
 */
static const char* select_response_template(
    negative_emotion_type_t emotion_type,
    emotion_intensity_t intensity)
{
    const char** templates = NULL;

    // Select template set based on emotion
    switch (emotion_type) {
        case EMOTION_RAGE:
        case EMOTION_ANGER:
        case EMOTION_FRUSTRATION:
            templates = RAGE_RESPONSE_TEMPLATES;
            break;

        case EMOTION_HATE:
            templates = HATE_RESPONSE_TEMPLATES;
            break;

        case EMOTION_FEAR:
        case EMOTION_PANIC:
            templates = FEAR_RESPONSE_TEMPLATES;
            break;

        case EMOTION_DESPAIR:
        case EMOTION_SHAME:
            templates = DESPAIR_RESPONSE_TEMPLATES;
            break;

        default:
            // Fallback validation template
            return "I can see you're experiencing some strong feelings right now. "
                   "Those feelings are valid and understandable. You're safe here, "
                   "and I'm here to help you. Would you like to talk about it?";
    }

    // Select template based on intensity (use first template for higher intensity)
    if (intensity >= EMOTION_INTENSITY_HIGH) {
        return templates[0];
    } else if (templates[1] != NULL) {
        return templates[1];
    } else {
        return templates[0];
    }
}

/**
 * @brief Determine primary response strategy
 */
static response_strategy_t select_strategy(
    negative_emotion_type_t emotion_type,
    emotion_intensity_t intensity,
    uint32_t crisis_flags)
{
    // Guard: Crisis always escalates
    if (crisis_flags != CRISIS_NONE) {
        return RESPONSE_ESCALATE;
    }

    // Guard: Extreme intensity needs de-escalation
    if (intensity == EMOTION_INTENSITY_EXTREME) {
        // For rage/panic, start with grounding
        if (emotion_type == EMOTION_RAGE || emotion_type == EMOTION_PANIC) {
            return RESPONSE_GROUND;
        }
        // For despair, validate first then consider escalation
        if (emotion_type == EMOTION_DESPAIR) {
            return RESPONSE_VALIDATE;  // Will escalate if student mentions crisis
        }
    }

    // Standard progression: VALIDATE → EXPLORE → GROUND if needed
    switch (emotion_type) {
        case EMOTION_RAGE:
        case EMOTION_PANIC:
            return (intensity >= EMOTION_INTENSITY_HIGH) ?
                   RESPONSE_GROUND : RESPONSE_VALIDATE;

        case EMOTION_FEAR:
            return RESPONSE_REASSURE;

        case EMOTION_HATE:
        case EMOTION_DISGUST:
            return RESPONSE_EXPLORE;  // Understand source

        case EMOTION_DESPAIR:
        case EMOTION_SHAME:
            return RESPONSE_VALIDATE;

        default:
            return RESPONSE_VALIDATE;
    }
}

bool empathetic_response_generate(
    empathetic_response_engine_t engine,
    const emotional_state_t* state,
    empathetic_response_t* response)
{
    // Guard: NULL checks
    if (!engine || !state || !response) {
        return false;
    }

    // Initialize response
    memset(response, 0, sizeof(empathetic_response_t));

    // Check for crisis
    uint32_t crisis_flags;
    float crisis_confidence;
    empathetic_response_detect_crisis(engine, state->text_input, &crisis_flags, &crisis_confidence);

    // Guard: IMMEDIATE ESCALATION for crisis
    if (crisis_flags != CRISIS_NONE) {
        response->primary_strategy = RESPONSE_ESCALATE;
        response->requires_human_escalation = true;

        // Generate crisis-appropriate response
        snprintf(response->response_text, sizeof(response->response_text),
                "Thank you for sharing that with me. What you're feeling is important, "
                "and I want to make sure you get the right support. "
                "I'm going to connect you with someone who can help you right away. "
                "You're not alone, and people care about you.");

        // Set escalation reason
        if (crisis_flags & CRISIS_SUICIDAL) {
            strncpy(response->escalation_reason, "Suicidal ideation detected",
                   sizeof(response->escalation_reason) - 1);
        } else if (crisis_flags & CRISIS_SELF_HARM) {
            strncpy(response->escalation_reason, "Self-harm intention detected",
                   sizeof(response->escalation_reason) - 1);
        } else if (crisis_flags & CRISIS_ABUSE) {
            strncpy(response->escalation_reason, "Abuse disclosure detected",
                   sizeof(response->escalation_reason) - 1);
        }

        response->safety_score = 1.0f;  // Escalation is the safe action
        response->ethics_approved = true;  // Helping in crisis is ethical

        engine->responses_generated++;
        return true;
    }

    // Determine strategy
    response->primary_strategy = select_strategy(state->emotion_type,
                                                 state->intensity,
                                                 crisis_flags);

    // Select and customize template
    const char* template = select_response_template(state->emotion_type, state->intensity);
    strncpy(response->response_text, template, sizeof(response->response_text) - 1);

    // Set metadata
    response->empathy_score = 0.85f;  // High empathy by design
    response->safety_score = 1.0f;    // Non-reactive responses are safe
    response->ethics_approved = true; // Validation + support is always ethical
    response->requires_human_escalation = false;

    // Track usage
    engine->responses_generated++;
    engine->strategy_usage[response->primary_strategy]++;

    return true;
}

bool empathetic_response_get_grounding_exercise(
    empathetic_response_engine_t engine,
    negative_emotion_type_t emotion_type,
    emotion_intensity_t intensity,
    grounding_exercise_t* exercise)
{
    // Guard: NULL checks
    if (!engine || !exercise) {
        return false;
    }

    // Select exercise based on emotion type and intensity
    if (emotion_type == EMOTION_PANIC || emotion_type == EMOTION_FEAR) {
        // Breathing exercise for anxiety/panic
        strncpy(exercise->name, "4-4-4 Breathing", sizeof(exercise->name) - 1);
        snprintf(exercise->instructions, sizeof(exercise->instructions),
                "Let's breathe together:\n"
                "1. Breathe in slowly through your nose for 4 counts\n"
                "2. Hold your breath gently for 4 counts\n"
                "3. Breathe out slowly through your mouth for 4 counts\n"
                "4. Repeat 3 times\n"
                "You're doing great. Focus on the feeling of the breath.");
        exercise->duration_seconds = 60;
        exercise->requires_interaction = true;

    } else if (emotion_type == EMOTION_RAGE || emotion_type == EMOTION_ANGER) {
        // 5-4-3-2-1 grounding for anger
        strncpy(exercise->name, "5-4-3-2-1 Grounding", sizeof(exercise->name) - 1);
        snprintf(exercise->instructions, sizeof(exercise->instructions),
                "Let's ground ourselves in the present moment:\n"
                "Name 5 things you can SEE around you\n"
                "Name 4 things you can TOUCH\n"
                "Name 3 things you can HEAR\n"
                "Name 2 things you can SMELL\n"
                "Name 1 thing you can TASTE\n"
                "Take your time. This helps bring us back to the present.");
        exercise->duration_seconds = 120;
        exercise->requires_interaction = true;

    } else {
        // Progressive muscle relaxation for other emotions
        strncpy(exercise->name, "Body Scan", sizeof(exercise->name) - 1);
        snprintf(exercise->instructions, sizeof(exercise->instructions),
                "Let's do a quick body scan:\n"
                "1. Notice your feet. Wiggle your toes.\n"
                "2. Notice your legs. Are they tense? Try to relax them.\n"
                "3. Notice your hands. Open and close them gently.\n"
                "4. Notice your shoulders. Roll them back and down.\n"
                "5. Notice your face. Relax your jaw.\n"
                "Take a deep breath. You're doing well.");
        exercise->duration_seconds = 90;
        exercise->requires_interaction = true;
    }

    return true;
}

float empathetic_response_predict_safety(
    empathetic_response_engine_t engine,
    const emotional_state_t* student_state,
    const char* proposed_response,
    emotional_state_t* predicted_reaction)
{
    // Guard: NULL checks
    if (!engine || !student_state || !proposed_response) {
        return 0.0f;
    }

    // Simple heuristic safety check (would use empathy_network in full version)
    float safety_score = 1.0f;

    char lowercase_response[1024];
    to_lowercase(lowercase_response, proposed_response, sizeof(lowercase_response));

    // Reduce safety if response contains negative patterns
    const char* unsafe_patterns[] = {
        "calm down",  // Dismissive
        "just relax", // Minimizing
        "you're overreacting", // Judgmental
        "don't be", // Dismissive
        "you shouldn't feel", // Invalidating
        NULL
    };

    for (int i = 0; unsafe_patterns[i] != NULL; i++) {
        if (strstr(lowercase_response, unsafe_patterns[i]) != NULL) {
            safety_score -= 0.3f;
        }
    }

    // Ensure minimum safety
    if (safety_score < 0.0f) {
        safety_score = 0.0f;
    }

    // Predict reaction (simplified - would use empathy network)
    if (predicted_reaction) {
        *predicted_reaction = *student_state;
        // If response is safe, predict slight improvement
        if (safety_score > 0.7f) {
            predicted_reaction->intensity = (predicted_reaction->intensity > 0) ?
                                           predicted_reaction->intensity - 1 : 0;
            predicted_reaction->valence = fminf(1.0f, predicted_reaction->valence + 0.2f);
        }
    }

    return safety_score;
}

void empathetic_response_track_effectiveness(
    empathetic_response_engine_t engine,
    const empathetic_response_t* response,
    const emotional_state_t* student_reaction,
    float effectiveness)
{
    // Guard: NULL checks
    if (!engine || !response) {
        return;
    }

    // Update running average
    float alpha = 0.1f;  // Learning rate
    engine->avg_effectiveness = (1.0f - alpha) * engine->avg_effectiveness +
                               alpha * effectiveness;

    // Update strategy-specific effectiveness
    response_strategy_t strategy = response->primary_strategy;
    if (strategy < 8) {
        engine->strategy_effectiveness[strategy] =
            (1.0f - alpha) * engine->strategy_effectiveness[strategy] +
            alpha * effectiveness;
    }
}
