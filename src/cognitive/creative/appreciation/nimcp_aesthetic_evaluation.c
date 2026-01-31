//=============================================================================
// nimcp_aesthetic_evaluation.c - Aesthetic Quality Evaluation Implementation
//=============================================================================
/**
 * @file nimcp_aesthetic_evaluation.c
 * @brief Implementation of aesthetic quality evaluation
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-01-30
 */

#include "cognitive/creative/appreciation/nimcp_aesthetic_evaluation.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "AESTHETIC_EVAL"

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for aesthetic_evaluation module */
static nimcp_health_agent_t* g_aesthetic_evaluation_health_agent = NULL;

/**
 * @brief Set health agent for aesthetic_evaluation heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void aesthetic_evaluation_set_health_agent(nimcp_health_agent_t* agent) {
    g_aesthetic_evaluation_health_agent = agent;
}

/** @brief Send heartbeat from aesthetic_evaluation module */
static inline void aesthetic_evaluation_heartbeat(const char* operation, float progress) {
    if (g_aesthetic_evaluation_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_aesthetic_evaluation_health_agent, operation, progress);
    }
}

//=============================================================================
// Configuration Defaults
//=============================================================================

void aesthetic_evaluator_config_defaults(aesthetic_evaluator_config_t* config) {
    if (!config) return;

    /* Dimension weights (balanced) */
    config->novelty_weight = 0.2f;
    config->complexity_weight = 0.2f;
    config->familiarity_weight = 0.2f;
    config->hedonic_weight = 0.2f;
    config->arousal_weight = 0.2f;

    /* Quality thresholds */
    config->min_quality_threshold = 0.3f;
    config->max_complexity_preference = 0.7f;
    config->optimal_novelty = 0.6f;

    /* Enable all modalities */
    config->enable_text_analysis = true;
    config->enable_music_analysis = true;
    config->enable_visual_analysis = true;

    /* Integration flags */
    config->use_emotion_system = false;
    config->use_memory_system = false;

    /* Resource limits */
    config->max_analysis_time_ms = 1000;
}

//=============================================================================
// Lifecycle API
//=============================================================================

aesthetic_evaluator_t* aesthetic_evaluator_create(
    const aesthetic_evaluator_config_t* config) {

    aesthetic_evaluator_t* eval = nimcp_calloc(1, sizeof(aesthetic_evaluator_t));
    if (!eval) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate aesthetic evaluator");
        return NULL;
    }

    /* Apply config */
    if (config) {
        memcpy(&eval->config, config, sizeof(aesthetic_evaluator_config_t));
    } else {
        aesthetic_evaluator_config_defaults(&eval->config);
    }

    /* Initialize statistics */
    eval->total_evaluations = 0;
    eval->avg_quality_score = 0.0f;
    eval->last_evaluation_us = 0;

    LOG_INFO(LOG_MODULE, "Aesthetic evaluator created");
    return eval;
}

void aesthetic_evaluator_destroy(aesthetic_evaluator_t* eval) {
    if (!eval) return;

    /* Free any allocated models - currently stubs */

    nimcp_free(eval);
    LOG_INFO(LOG_MODULE, "Aesthetic evaluator destroyed");
}

//=============================================================================
// Evaluation API - Stub implementations
//=============================================================================

int aesthetic_evaluate_text(aesthetic_evaluator_t* eval,
                            const char* text, size_t len,
                            art_modality_t modality,
                            aesthetic_evaluation_t* out) {
    if (!eval || !text || !out) return -1;

    memset(out, 0, sizeof(aesthetic_evaluation_t));

    /* Stub: set reasonable defaults for now */
    out->dimensions.novelty = 0.5f;
    out->dimensions.complexity = 0.5f;
    out->dimensions.familiarity = 0.5f;
    out->dimensions.hedonic_tone = 0.0f;
    out->dimensions.arousal_potential = 0.5f;

    /* Calculate weighted score */
    float score = 0.0f;
    score += out->dimensions.novelty * eval->config.novelty_weight;
    score += out->dimensions.complexity * eval->config.complexity_weight;
    score += out->dimensions.familiarity * eval->config.familiarity_weight;
    score += (out->dimensions.hedonic_tone + 1.0f) / 2.0f * eval->config.hedonic_weight;
    score += out->dimensions.arousal_potential * eval->config.arousal_weight;

    /* Update statistics */
    eval->total_evaluations++;
    eval->avg_quality_score = (eval->avg_quality_score * (eval->total_evaluations - 1) + score) /
                               eval->total_evaluations;

    return 0;
}

int aesthetic_evaluate_music(aesthetic_evaluator_t* eval,
                              const music_track_t* tracks, uint32_t num_tracks,
                              aesthetic_evaluation_t* out) {
    if (!eval || !tracks || !out) return -1;

    memset(out, 0, sizeof(aesthetic_evaluation_t));

    /* Stub: set reasonable defaults */
    out->dimensions.novelty = 0.5f;
    out->dimensions.complexity = 0.5f;
    out->dimensions.familiarity = 0.5f;
    out->dimensions.hedonic_tone = 0.2f;
    out->dimensions.arousal_potential = 0.6f;

    eval->total_evaluations++;

    return 0;
}

int aesthetic_evaluate_visual(aesthetic_evaluator_t* eval,
                               const visual_image_t* image,
                               aesthetic_evaluation_t* out) {
    if (!eval || !image || !out) return -1;

    memset(out, 0, sizeof(aesthetic_evaluation_t));

    /* Stub: set reasonable defaults */
    out->dimensions.novelty = 0.5f;
    out->dimensions.complexity = 0.5f;
    out->dimensions.familiarity = 0.5f;
    out->dimensions.hedonic_tone = 0.3f;
    out->dimensions.arousal_potential = 0.5f;

    eval->total_evaluations++;

    return 0;
}

int aesthetic_evaluate(aesthetic_evaluator_t* eval,
                        const void* content, art_modality_t modality,
                        aesthetic_evaluation_t* out) {
    if (!eval || !content || !out) return -1;

    /* Route to appropriate modality-specific evaluator */
    switch (modality) {
        case ART_MODALITY_TEXT_PROSE:
        case ART_MODALITY_TEXT_POETRY:
        case ART_MODALITY_TEXT_SCREENPLAY:
        case ART_MODALITY_TEXT_LYRICS:
        case ART_MODALITY_TEXT_DIALOGUE:
            return aesthetic_evaluate_text(eval, (const char*)content,
                                          strlen((const char*)content), modality, out);

        case ART_MODALITY_VISUAL_PAINTING:
        case ART_MODALITY_VISUAL_DIGITAL:
        case ART_MODALITY_VISUAL_PHOTO:
        case ART_MODALITY_VISUAL_ILLUSTRATION:
        case ART_MODALITY_VISUAL_3D:
            return aesthetic_evaluate_visual(eval, (const visual_image_t*)content, out);

        case ART_MODALITY_MUSIC_CLASSICAL:
        case ART_MODALITY_MUSIC_FILM_SCORE:
        case ART_MODALITY_MUSIC_JAZZ:
        case ART_MODALITY_MUSIC_ELECTRONIC:
        case ART_MODALITY_MUSIC_FOLK:
            return aesthetic_evaluate_music(eval, (const music_track_t*)content, 1, out);

        default:
            LOG_WARN(LOG_MODULE, "Unsupported modality: %d", modality);
            return -1;
    }
}

//=============================================================================
// Integration API
//=============================================================================

void aesthetic_evaluator_set_emotion_system(aesthetic_evaluator_t* eval,
                                             void* emotion_system) {
    if (!eval) return;
    eval->emotion_system = emotion_system;
    eval->config.use_emotion_system = (emotion_system != NULL);
}

void aesthetic_evaluator_set_memory_system(aesthetic_evaluator_t* eval,
                                            void* memory_system) {
    if (!eval) return;
    eval->memory_system = memory_system;
    eval->config.use_memory_system = (memory_system != NULL);
}

void aesthetic_evaluator_set_visual_cortex(aesthetic_evaluator_t* eval,
                                            void* visual_cortex) {
    if (!eval) return;
    eval->visual_cortex = visual_cortex;
}

void aesthetic_evaluator_set_audio_cortex(aesthetic_evaluator_t* eval,
                                           void* audio_cortex) {
    if (!eval) return;
    eval->audio_cortex = audio_cortex;
}

void aesthetic_evaluator_set_speech_cortex(aesthetic_evaluator_t* eval,
                                            void* speech_cortex) {
    if (!eval) return;
    eval->speech_cortex = speech_cortex;
}

//=============================================================================
// Statistics API
//=============================================================================

uint64_t aesthetic_evaluator_get_total_evaluations(const aesthetic_evaluator_t* eval) {
    return eval ? eval->total_evaluations : 0;
}

float aesthetic_evaluator_get_avg_quality(const aesthetic_evaluator_t* eval) {
    return eval ? eval->avg_quality_score : 0.0f;
}

void aesthetic_evaluator_reset_stats(aesthetic_evaluator_t* eval) {
    if (!eval) return;
    eval->total_evaluations = 0;
    eval->avg_quality_score = 0.0f;
}
