//=============================================================================
// nimcp_creative.c - Creative System Core Implementation
//=============================================================================
/**
 * @file nimcp_creative.c
 * @brief Core implementation for creative/artistic cognitive system
 *
 * WHAT: Core types, utilities, and initialization for creative system
 * WHY:  Provide foundation for artistic appreciation and generation
 * HOW:  Implements shared utilities used across creative subsystems
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-01-30
 */

#include "cognitive/creative/nimcp_creative.h"
#include "cognitive/creative/generation/nimcp_video_generation.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "CREATIVE"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"
#include "utils/exception/nimcp_exception_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(creative)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_creative_mesh_id = 0;
static mesh_participant_registry_t* g_creative_mesh_registry = NULL;

nimcp_error_t creative_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_creative_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "creative", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "creative";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_creative_mesh_id);
    if (err == NIMCP_SUCCESS) g_creative_mesh_registry = registry;
    return err;
}

void creative_mesh_unregister(void) {
    if (g_creative_mesh_registry && g_creative_mesh_id != 0) {
        mesh_participant_unregister(g_creative_mesh_registry, g_creative_mesh_id);
        g_creative_mesh_id = 0;
        g_creative_mesh_registry = NULL;
    }
}


/** @brief Dual-level heartbeat: instance agent or global */
static inline void creative_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (instance_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    } else if (g_creative_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_creative_health_agent, operation, progress);
    }
}

//=============================================================================
// Config Defaults
//=============================================================================

void creative_config_init_defaults(creative_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(creative_config_t));

    /* Enable all subsystems by default */
    config->enable_appreciation = true;
    config->enable_inspiration = true;
    config->enable_text_generation = true;
    config->enable_music_generation = true;
    config->enable_visual_generation = true;
    config->enable_video_generation = true;
    config->enable_multimodal_direction = true;

    /* Quality defaults */
    config->min_quality_threshold = 0.7f;
    config->copyright_similarity_threshold = 0.85f;

    /* Resource limits */
    config->max_generation_time_ms = 60000;  /* 1 minute */
    config->max_memory_bytes = 8ULL * 1024 * 1024 * 1024;  /* 8GB */

    /* Device */
    config->device_type = 0;  /* CPU default */
    config->device_id = 0;

    /* Integration flags */
    config->integrate_with_emotion = true;
    config->integrate_with_memory = true;
    config->integrate_with_ethics = true;
    config->integrate_with_immune = true;

    /* API fallback */
    config->use_cloud_fallback = true;
}

//=============================================================================
// Modality Utilities
//=============================================================================

const char* art_modality_name(art_modality_t modality) {
    switch (modality) {
        case ART_MODALITY_TEXT_POETRY:       return "poetry";
        case ART_MODALITY_TEXT_PROSE:        return "prose";
        case ART_MODALITY_TEXT_SCREENPLAY:   return "screenplay";
        case ART_MODALITY_TEXT_LYRICS:       return "lyrics";
        case ART_MODALITY_TEXT_DIALOGUE:     return "dialogue";
        case ART_MODALITY_MUSIC_CLASSICAL:   return "classical_music";
        case ART_MODALITY_MUSIC_FILM_SCORE:  return "film_score";
        case ART_MODALITY_MUSIC_JAZZ:        return "jazz";
        case ART_MODALITY_MUSIC_ELECTRONIC:  return "electronic";
        case ART_MODALITY_MUSIC_FOLK:        return "folk";
        case ART_MODALITY_VISUAL_PAINTING:   return "painting";
        case ART_MODALITY_VISUAL_DIGITAL:    return "digital_art";
        case ART_MODALITY_VISUAL_PHOTO:      return "photography";
        case ART_MODALITY_VISUAL_ILLUSTRATION: return "illustration";
        case ART_MODALITY_VISUAL_3D:         return "3d_render";
        case ART_MODALITY_VIDEO_CINEMA:      return "cinema";
        case ART_MODALITY_VIDEO_ANIMATION:   return "animation";
        case ART_MODALITY_VIDEO_DOCUMENTARY: return "documentary";
        case ART_MODALITY_VIDEO_MUSIC_VIDEO: return "music_video";
        case ART_MODALITY_VIDEO_SHORT:       return "short_film";
        default:                             return "unknown";
    }
}

//=============================================================================
// Aesthetic Dimensions Utilities
//=============================================================================

void aesthetic_dimensions_init(aesthetic_dimensions_t* dims) {
    if (!dims) return;
    dims->novelty = 0.5f;
    dims->complexity = 0.5f;
    dims->familiarity = 0.5f;
    dims->hedonic_tone = 0.0f;
    dims->arousal_potential = 0.5f;
}

float aesthetic_dimensions_score(const aesthetic_dimensions_t* dims) {
    if (!dims) return 0.0f;

    /* Berlyne's optimal arousal theory: medium complexity/novelty is preferred */
    /* Peak aesthetic preference at moderate levels */
    float novelty_contrib = 1.0f - fabsf(dims->novelty - 0.5f) * 2.0f;
    float complexity_contrib = 1.0f - fabsf(dims->complexity - 0.6f) * 2.0f;
    float familiarity_contrib = dims->familiarity * 0.5f + 0.25f;

    /* Hedonic tone directly contributes (rescaled from [-1,1] to [0,1]) */
    float hedonic_contrib = (dims->hedonic_tone + 1.0f) * 0.5f;

    /* Arousal contributes with preference for moderate levels */
    float arousal_contrib = 1.0f - fabsf(dims->arousal_potential - 0.5f) * 1.5f;

    /* Weighted combination */
    float score = novelty_contrib * 0.2f +
                  complexity_contrib * 0.25f +
                  familiarity_contrib * 0.15f +
                  hedonic_contrib * 0.25f +
                  arousal_contrib * 0.15f;

    /* Clamp to [0, 1] */
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;

    return score;
}

//=============================================================================
// Emotional Response Utilities
//=============================================================================

void aesthetic_emotional_response_init(aesthetic_emotional_response_t* response) {
    if (!response) return;
    memset(response, 0, sizeof(aesthetic_emotional_response_t));
}

float aesthetic_emotional_response_valence(const aesthetic_emotional_response_t* response) {
    if (!response) return 0.0f;

    /* Positive emotions */
    float positive = response->joy + response->trust + response->awe +
                     response->sublime + response->anticipation;

    /* Negative emotions */
    float negative = response->fear + response->sadness + response->anger +
                     response->disgust;

    /* Normalize */
    float total = positive + negative;
    if (total < 0.001f) return 0.0f;

    return (positive - negative) / total;
}

float aesthetic_emotional_response_arousal(const aesthetic_emotional_response_t* response) {
    if (!response) return 0.0f;

    /* High arousal emotions */
    float high = response->joy + response->fear + response->anger +
                 response->surprise + response->awe + response->anticipation;

    /* Low arousal emotions */
    float low = response->sadness + response->trust + response->contemplation;

    float total = high + low;
    if (total < 0.001f) return 0.5f;

    return high / total;
}

//=============================================================================
// Style Embedding Utilities
//=============================================================================

int style_embedding_create(style_embedding_t* embedding, uint32_t dim) {
    if (!embedding || dim == 0 || dim > 2048) {
        LOG_ERROR(LOG_MODULE, "Invalid embedding or dimension: %u", dim);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "style_embedding_create: embedding is NULL");
        return -1;
    }

    creative_heartbeat("style_embedding_create", 0.0f);

    memset(embedding, 0, sizeof(style_embedding_t));

    embedding->embedding = nimcp_calloc(dim, sizeof(float));
    if (!embedding->embedding) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate embedding values");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "style_embedding_create: embedding->embedding is NULL");
        return -1;
    }

    embedding->embedding_dim = dim;
    embedding->archetype_id = -1;  /* No archetype */
    embedding->confidence = 0.0f;

    creative_heartbeat("style_embedding_create", 1.0f);

    return 0;
}

void style_embedding_destroy(style_embedding_t* embedding) {
    if (!embedding) return;

    if (embedding->embedding) {
        nimcp_free(embedding->embedding);
        embedding->embedding = NULL;
    }
    embedding->embedding_dim = 0;
}

int style_embedding_clone(const style_embedding_t* src, style_embedding_t* dst) {
    if (!src || !dst) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "style_embedding_clone: required parameter is NULL (src, dst)");
        return -1;
    }

    if (style_embedding_create(dst, src->embedding_dim) != 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "style_embedding_clone: validation failed");
        return -1;
    }

    memcpy(dst->embedding, src->embedding, src->embedding_dim * sizeof(float));
    dst->modality = src->modality;
    dst->archetype_id = src->archetype_id;
    memcpy(dst->style_name, src->style_name, sizeof(dst->style_name));
    dst->confidence = src->confidence;

    return 0;
}

float style_embedding_similarity(const style_embedding_t* a,
                                  const style_embedding_t* b) {
    if (!a || !b || a->embedding_dim != b->embedding_dim || a->embedding_dim == 0) {
        return 0.0f;
    }

    /* Cosine similarity */
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t i = 0; i < a->embedding_dim; i++) {
        dot += a->embedding[i] * b->embedding[i];
        norm_a += a->embedding[i] * a->embedding[i];
        norm_b += b->embedding[i] * b->embedding[i];
    }

    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return 0.0f;
    }

    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

int style_embedding_interpolate(const style_embedding_t* a,
                                 const style_embedding_t* b,
                                 float t,
                                 style_embedding_t* result) {
    if (!a || !b || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "style_embedding_interpolate: required parameter is NULL (a, b, result)");
        return -1;
    }
    if (a->embedding_dim != b->embedding_dim || a->embedding_dim != result->embedding_dim) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "style_embedding_interpolate: dimension mismatch");
        return -1;
    }

    /* Clamp t to [0, 1] */
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    /* Linear interpolation */
    for (uint32_t i = 0; i < a->embedding_dim; i++) {
        result->embedding[i] = a->embedding[i] * (1.0f - t) + b->embedding[i] * t;
    }

    /* Interpolate archetype weight if same archetype */
    if (a->archetype_id == b->archetype_id) {
        result->archetype_id = a->archetype_id;
        result->confidence = a->confidence * (1.0f - t) +
                                   b->confidence * t;
    } else {
        result->archetype_id = -1;
        result->confidence = 0.0f;
    }

    return 0;
}

void style_embedding_normalize(style_embedding_t* embedding) {
    if (!embedding || !embedding->embedding || embedding->embedding_dim == 0) return;

    float norm = 0.0f;
    for (uint32_t i = 0; i < embedding->embedding_dim; i++) {
        norm += embedding->embedding[i] * embedding->embedding[i];
    }

    if (norm < 1e-8f) return;

    norm = sqrtf(norm);
    for (uint32_t i = 0; i < embedding->embedding_dim; i++) {
        embedding->embedding[i] /= norm;
    }
}

//=============================================================================
// Archetype Name Utilities
//=============================================================================

const char* literary_style_archetype_name(literary_style_archetype_t archetype) {
    switch (archetype) {
        case STYLE_LIT_HEMINGWAY:    return "Hemingway";
        case STYLE_LIT_TOLSTOY:      return "Tolstoy";
        case STYLE_LIT_JOYCE:        return "Joyce";
        case STYLE_LIT_POE:          return "Poe";
        case STYLE_LIT_AUSTEN:       return "Austen";
        case STYLE_LIT_SHAKESPEARE:  return "Shakespeare";
        case STYLE_LIT_BORGES:       return "Borges";
        case STYLE_LIT_KAFKA:        return "Kafka";
        case STYLE_LIT_MARQUEZ:      return "Marquez";
        case STYLE_LIT_DOSTOEVSKY:      return "Dostoevsky";
        case STYLE_LIT_WOOLF:        return "Woolf";
        case STYLE_LIT_FAULKNER:     return "Faulkner";
        default:                     return "Unknown";
    }
}

const char* musical_style_archetype_name(musical_style_archetype_t archetype) {
    switch (archetype) {
        case STYLE_MUSIC_BACH:           return "Bach";
        case STYLE_MUSIC_BEETHOVEN:      return "Beethoven";
        case STYLE_MUSIC_DEBUSSY:        return "Debussy";
        case STYLE_MUSIC_JOHN_WILLIAMS:  return "John Williams";
        case STYLE_MUSIC_MILES_DAVIS:    return "Miles Davis";
        case STYLE_MUSIC_HANS_ZIMMER:    return "Hans Zimmer";
        case STYLE_MUSIC_STRAVINSKY:     return "Stravinsky";
        case STYLE_MUSIC_ENNIO_MORRICONE: return "Ennio Morricone";
        case STYLE_MUSIC_SAKAMOTO:       return "Sakamoto";
        case STYLE_MUSIC_GLASS:          return "Philip Glass";
        case STYLE_MUSIC_COPLAND:        return "Copland";
        case STYLE_MUSIC_RAVEL:          return "Ravel";
        default:                         return "Unknown";
    }
}

const char* visual_style_archetype_name(visual_style_archetype_t archetype) {
    switch (archetype) {
        case STYLE_VIS_VAN_GOGH:     return "Van Gogh";
        case STYLE_VIS_MONET:        return "Monet";
        case STYLE_VIS_PICASSO:      return "Picasso";
        case STYLE_VIS_DALI:         return "Dali";
        case STYLE_VIS_REMBRANDT:    return "Rembrandt";
        case STYLE_VIS_WARHOL:       return "Warhol";
        case STYLE_VIS_KLIMT:        return "Klimt";
        case STYLE_VIS_HOKUSAI:      return "Hokusai";
        case STYLE_VIS_KANDINSKY:    return "Kandinsky";
        case STYLE_VIS_ESCHER:       return "Escher";
        case STYLE_VIS_BASQUIAT:     return "Basquiat";
        case STYLE_VIS_CARAVAGGIO:   return "Caravaggio";
        default:                     return "Unknown";
    }
}

const char* cinematic_style_archetype_name(cinematic_style_archetype_t archetype) {
    switch (archetype) {
        case STYLE_CINEMA_KUBRICK:     return "Kubrick";
        case STYLE_CINEMA_SPIELBERG:   return "Spielberg";
        case STYLE_CINEMA_TARANTINO:   return "Tarantino";
        case STYLE_CINEMA_NOLAN:       return "Nolan";
        case STYLE_CINEMA_TARKOVSKY:   return "Tarkovsky";
        case STYLE_CINEMA_MIYAZAKI:    return "Miyazaki";
        case STYLE_CINEMA_WELLES:      return "Welles";
        case STYLE_CINEMA_HITCHCOCK:   return "Hitchcock";
        case STYLE_CINEMA_COPPOLA:     return "Coppola";
        case STYLE_CINEMA_FINCHER:     return "Fincher";
        case STYLE_CINEMA_KUROSAWA:    return "Kurosawa";
        case STYLE_CINEMA_VILLENEUVE: return "Denis Villeneuve";
        default:                       return "Unknown";
    }
}

//=============================================================================
// Validation Result Utilities
//=============================================================================

const char* creative_validation_result_name(creative_validation_result_t result) {
    switch (result) {
        case CREATIVE_VALIDATION_PASS:     return "PASS";
        case CREATIVE_VALIDATION_WARN:     return "WARN";
        case CREATIVE_VALIDATION_ESCALATE: return "ESCALATE";
        case CREATIVE_VALIDATION_DENY:     return "DENY";
        default:                           return "UNKNOWN";
    }
}

const char* creative_deny_reason_name(creative_deny_reason_t reason) {
    switch (reason) {
        case CREATIVE_DENY_NONE:            return "none";
        case CREATIVE_DENY_COPYRIGHT:       return "copyright";
        case CREATIVE_DENY_HARMFUL_CONTENT: return "harmful_content";
        case CREATIVE_DENY_QUALITY:         return "low_quality";
        case CREATIVE_DENY_INCOHERENT:      return "incoherent";
        case CREATIVE_DENY_EXPLICIT:        return "explicit_content";
        case CREATIVE_DENY_BIAS:            return "bias_detected";
        case CREATIVE_DENY_MISINFORMATION:  return "misinformation";
        default:                            return "unknown";
    }
}

//=============================================================================
// Visual Image Utilities
//=============================================================================

visual_image_t* visual_image_create(uint32_t width, uint32_t height,
                                     uint32_t channels) {
    if (width == 0 || height == 0 || channels == 0 || channels > 4) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "visual_image_create: invalid dimensions or channels");
        return NULL;
    }

    /* Check for overflow: width * height */
    if (width > 0 && height > SIZE_MAX / width) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "visual_image_create: width*height overflow");
        return NULL;
    }
    size_t pixel_count = (size_t)width * height;
    if (channels > 0 && pixel_count > SIZE_MAX / channels) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "visual_image_create: pixel_count*channels overflow");
        return NULL;
    }
    size_t data_size = pixel_count * channels;

    visual_image_t* image = nimcp_calloc(1, sizeof(visual_image_t));
    if (!image) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "visual_image_create: image allocation failed");
        return NULL;
    }

    image->pixels = nimcp_calloc(data_size, sizeof(float));
    if (!image->pixels) {
        nimcp_free(image);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "visual_image_create: pixels allocation failed");
        return NULL;
    }

    image->width = width;
    image->height = height;
    image->channels = channels;

    return image;
}

void visual_image_destroy(visual_image_t* image) {
    if (!image) return;

    if (image->pixels) {
        nimcp_free(image->pixels);
    }
    nimcp_free(image);
}

visual_image_t* visual_image_clone(const visual_image_t* src) {
    if (!src) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "visual_image_clone: src is NULL");
        return NULL;
    }

    visual_image_t* dst = visual_image_create(src->width, src->height,
                                               src->channels);
    if (!dst) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "visual_image_clone: dst is NULL");
        return NULL;
    }

    /* P1-COG-01: Use dst dimensions (known-safe after successful create)
     * to avoid trusting potentially-corrupted src dimensions for memcpy size */
    size_t data_size = (size_t)dst->width * dst->height * dst->channels;
    memcpy(dst->pixels, src->pixels, data_size * sizeof(float));

    return dst;
}

//=============================================================================
// Music Track Utilities
//=============================================================================

/** P3-COG-02: Named constant for default track capacity */
#define CREATIVE_DEFAULT_MAX_NOTES 10000

music_track_t* music_track_create(uint32_t max_notes) {
    if (max_notes == 0) max_notes = CREATIVE_DEFAULT_MAX_NOTES;

    music_track_t* track = nimcp_calloc(1, sizeof(music_track_t));
    if (!track) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "music_track_create: track is NULL");
        return NULL;
    }

    track->notes = nimcp_calloc(max_notes, sizeof(music_note_t));
    if (!track->notes) {
        nimcp_free(track);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "music_track_create: track->notes is NULL");
        return NULL;
    }

    track->num_notes = 0;
    track->max_notes = max_notes;
    track->channel = 0;
    track->instrument = 0;
    strncpy(track->track_name, "Track", sizeof(track->track_name) - 1);

    return track;
}

void music_track_destroy(music_track_t* track) {
    if (!track) return;

    if (track->notes) {
        nimcp_free(track->notes);
    }
    nimcp_free(track);
}

int music_track_add_note(music_track_t* track, const music_note_t* note) {
    if (!track || !note) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "music_track_add_note: required parameter is NULL (track, note)");
        return -1;
    }

    /* Check capacity before adding */
    if (track->num_notes >= track->max_notes) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "music_track_add_note: track is full");
        return -1;
    }

    track->notes[track->num_notes] = *note;
    track->num_notes++;

    return 0;
}

//=============================================================================
// Generation Result Free Functions
//=============================================================================

void text_generation_result_free(text_generation_result_t* result) {
    if (!result) return;

    if (result->text) {
        nimcp_free(result->text);
        result->text = NULL;
    }
    result->text_len = 0;
}

void music_generation_result_free(music_generation_result_t* result) {
    if (!result) return;

    if (result->tracks) {
        for (uint32_t i = 0; i < result->num_tracks; i++) {
            if (result->tracks[i].notes) {
                nimcp_free(result->tracks[i].notes);
            }
        }
        nimcp_free(result->tracks);
        result->tracks = NULL;
    }

    if (result->audio_data) {
        nimcp_free(result->audio_data);
        result->audio_data = NULL;
    }

    result->num_tracks = 0;
    result->audio_samples = 0;
}

void visual_generation_result_free(visual_generation_result_t* result) {
    if (!result) return;

    if (result->image.pixels) {
        nimcp_free(result->image.pixels);
        result->image.pixels = NULL;
    }
}

/* video_generation_result_free is in nimcp_video_generation.c */

//=============================================================================
// Aliased Result Free Functions (for API consistency)
//=============================================================================

void creative_text_result_free(text_generation_result_t* result) {
    text_generation_result_free(result);
}

void creative_music_result_free(music_generation_result_t* result) {
    music_generation_result_free(result);
}

void creative_visual_result_free(visual_generation_result_t* result) {
    visual_generation_result_free(result);
}

void creative_project_spec_free(project_specification_t* spec) {
    if (!spec) return;

    /* Free dynamically allocated scene data */
    if (spec->scenes) {
        for (uint32_t i = 0; i < spec->num_scenes; i++) {
            if (spec->scenes[i].dialogue) {
                nimcp_free(spec->scenes[i].dialogue);
                spec->scenes[i].dialogue = NULL;
            }
            if (spec->scenes[i].keyframes) {
                for (uint32_t j = 0; j < spec->scenes[i].num_keyframes; j++) {
                    if (spec->scenes[i].keyframes[j].pixels &&
                        spec->scenes[i].keyframes[j].owns_pixels) {
                        nimcp_free(spec->scenes[i].keyframes[j].pixels);
                    }
                }
                nimcp_free(spec->scenes[i].keyframes);
                spec->scenes[i].keyframes = NULL;
            }
            if (spec->scenes[i].music_cue) {
                if (spec->scenes[i].music_cue->notes) {
                    nimcp_free(spec->scenes[i].music_cue->notes);
                }
                nimcp_free(spec->scenes[i].music_cue);
                spec->scenes[i].music_cue = NULL;
            }
        }
        nimcp_free(spec->scenes);
        spec->scenes = NULL;
    }
    spec->num_scenes = 0;

    /* Free style embeddings */
    style_embedding_destroy(&spec->visual_style);
    style_embedding_destroy(&spec->music_style);
}

void creative_project_output_free(project_output_t* output) {
    if (!output) return;

    if (output->video_data) {
        nimcp_free(output->video_data);
        output->video_data = NULL;
    }
    output->video_size = 0;

    /* Free embedded result resources */
    music_generation_result_free(&output->soundtrack);
    text_generation_result_free(&output->screenplay);
}

void creative_style_embedding_free(style_embedding_t* style) {
    if (!style) return;
    style_embedding_destroy(style);
}

void creative_blend_result_free(influence_blend_result_t* result) {
    if (!result) return;

    /* Free blended style embedding */
    style_embedding_destroy(&result->blended_style);

    /* Free influence weights if allocated */
    if (result->influence_weights) {
        nimcp_free(result->influence_weights);
        result->influence_weights = NULL;
    }

    result->num_influences = 0;
}

//=============================================================================
// Appreciation API - Stub implementations
//=============================================================================

int creative_evaluate_text(creative_orchestrator_t* orchestrator,
                           const char* text, size_t len,
                           art_modality_t modality,
                           aesthetic_evaluation_t* out) {
    if (!orchestrator || !text || !out) return -1;
    (void)len;
    (void)modality;
    memset(out, 0, sizeof(*out));
    return -1;  /* Not yet implemented */
}

int creative_evaluate_music(creative_orchestrator_t* orchestrator,
                            const music_track_t* tracks, uint32_t num_tracks,
                            aesthetic_evaluation_t* out) {
    if (!orchestrator || !tracks || !out) return -1;
    (void)num_tracks;
    memset(out, 0, sizeof(*out));
    return -1;  /* Not yet implemented */
}

int creative_evaluate_visual(creative_orchestrator_t* orchestrator,
                             const visual_image_t* image,
                             aesthetic_evaluation_t* out) {
    if (!orchestrator || !image || !out) return -1;
    memset(out, 0, sizeof(*out));
    return -1;  /* Not yet implemented */
}

//=============================================================================
// Inspiration API - Stub implementations
//=============================================================================

int creative_extract_style(creative_orchestrator_t* orchestrator,
                           const void* content,
                           art_modality_t modality,
                           style_embedding_t* out) {
    if (!orchestrator || !content || !out) return -1;
    (void)modality;
    memset(out, 0, sizeof(*out));
    return -1;  /* Not yet implemented */
}

int creative_blend_influences(creative_orchestrator_t* orchestrator,
                              const creative_influence_t* influences,
                              uint32_t num_influences,
                              influence_blend_result_t* out) {
    if (!orchestrator || !influences || !out) return -1;
    (void)num_influences;
    memset(out, 0, sizeof(*out));
    return -1;  /* Not yet implemented */
}

int creative_get_archetype_style(creative_orchestrator_t* orchestrator,
                                 art_modality_t modality,
                                 int32_t archetype_id,
                                 style_embedding_t* out) {
    if (!orchestrator || !out) return -1;
    (void)modality;
    (void)archetype_id;
    memset(out, 0, sizeof(*out));
    return -1;  /* Not yet implemented */
}

//=============================================================================
// Generation API - Stub implementations
//=============================================================================

int creative_generate_text(creative_orchestrator_t* orchestrator,
                           const text_generation_request_t* request,
                           text_generation_result_t* result) {
    if (!orchestrator || !request || !result) return -1;
    memset(result, 0, sizeof(*result));
    return -1;  /* Not yet implemented */
}

int creative_generate_music(creative_orchestrator_t* orchestrator,
                            const music_generation_request_t* request,
                            music_generation_result_t* result) {
    if (!orchestrator || !request || !result) return -1;
    memset(result, 0, sizeof(*result));
    return -1;  /* Not yet implemented */
}

int creative_generate_visual(creative_orchestrator_t* orchestrator,
                             const visual_generation_request_t* request,
                             visual_generation_result_t* result) {
    if (!orchestrator || !request || !result) return -1;
    memset(result, 0, sizeof(*result));
    return -1;  /* Not yet implemented */
}
