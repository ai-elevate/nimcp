//=============================================================================
// nimcp_style_representation.c - Style Embedding and Representation
//=============================================================================
/**
 * @file nimcp_style_representation.c
 * @brief Creates and manages style embeddings for creative inspiration
 *
 * WHAT: Generates dense vector representations of artistic styles
 * WHY:  Enable style transfer, blending, and comparison
 * HOW:  Neural embeddings + archetype-based parametric representations
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-01-30
 */

#include "cognitive/creative/inspiration/nimcp_style_representation.h"
#include "cognitive/creative/nimcp_creative.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "STYLE_REPR"

#define DEFAULT_EMBEDDING_DIM 256

//=============================================================================
// Static Archetype Data
//=============================================================================

/* Literary archetype info */
static const struct {
    const char* name;
    const char* description;
    const char* era;
    const char* characteristics;
} literary_archetype_data[] = {
    {"Hemingway", "Minimalist American prose master", "20th Century",
     "Iceberg theory, short sentences, masculine themes, understated emotion"},
    {"Tolstoy", "Russian epic realist", "19th Century",
     "Epic scope, psychological depth, moral philosophy, panoramic narratives"},
    {"Joyce", "Modernist experimentalist", "20th Century",
     "Stream of consciousness, Dublin, linguistic innovation, mythic parallels"},
    {"Poe", "Gothic master", "19th Century",
     "Gothic horror, psychological terror, unreliable narrators, dark poetry"},
    {"Austen", "Regency social observer", "19th Century",
     "Social comedy, romantic irony, free indirect discourse, wit"},
    {"Shakespeare", "Universal dramatist", "Renaissance",
     "Iambic pentameter, wordplay, universal themes, character depth"},
    {"Borges", "Metaphysical fabulist", "20th Century",
     "Labyrinths, infinite libraries, philosophical puzzles, metafiction"},
    {"Kafka", "Absurdist visionary", "20th Century",
     "Bureaucratic nightmares, alienation, transformation, paranoia"},
    {"Marquez", "Magical realist", "20th Century",
     "Magical realism, Macondo, multi-generational, mythic time"}
};

static const struct {
    const char* name;
    const char* description;
    const char* era;
    const char* characteristics;
} music_archetype_data[] = {
    {"Bach", "Baroque master", "Baroque",
     "Counterpoint, fugues, mathematical precision, sacred works"},
    {"Beethoven", "Romantic titan", "Classical/Romantic",
     "Heroic themes, symphonic development, emotional intensity"},
    {"Debussy", "Impressionist pioneer", "Late Romantic",
     "Impressionism, whole-tone scales, atmospheric textures"},
    {"John Williams", "Film score maestro", "Contemporary",
     "Leitmotifs, orchestral grandeur, heroic fanfares"},
    {"Miles Davis", "Jazz innovator", "Modern Jazz",
     "Modal jazz, cool jazz, improvisation, Kind of Blue"},
    {"Hans Zimmer", "Electronic-orchestral pioneer", "Contemporary",
     "Electronic-orchestral hybrid, pulsing rhythms, epic scale"}
};

static const struct {
    const char* name;
    const char* description;
    const char* era;
    const char* characteristics;
} visual_archetype_data[] = {
    {"Van Gogh", "Post-Impressionist", "Post-Impressionism",
     "Swirling brushwork, vivid colors, emotional intensity"},
    {"Monet", "Impressionist master", "Impressionism",
     "Light studies, water lilies, plein air, atmospheric"},
    {"Picasso", "Cubist revolutionary", "Modern",
     "Cubism, multiple perspectives, periods, abstraction"},
    {"Dali", "Surrealist icon", "Surrealism",
     "Melting clocks, dreamscapes, paranoid-critical method"},
    {"Warhol", "Pop art pioneer", "Pop Art",
     "Celebrity culture, mass production, screen printing"},
    {"Rembrandt", "Baroque master", "Baroque",
     "Chiaroscuro, psychological portraits, Dutch Golden Age"}
};

static const struct {
    const char* name;
    const char* description;
    const char* era;
    const char* characteristics;
} cinema_archetype_data[] = {
    {"Kubrick", "Perfectionist auteur", "Modern",
     "One-point perspective, cold beauty, humanity themes"},
    {"Spielberg", "Blockbuster master", "Contemporary",
     "Wonder, family themes, emotional storytelling"},
    {"Tarantino", "Postmodern stylist", "Contemporary",
     "Non-linear narrative, pop culture, stylized violence"},
    {"Nolan", "Cerebral blockbuster", "Contemporary",
     "Mind-bending plots, practical effects, time manipulation"},
    {"Tarkovsky", "Poetic cinema", "Art Film",
     "Long takes, spiritual themes, poetic imagery"},
    {"Miyazaki", "Animation master", "Contemporary",
     "Hand-drawn animation, flight, environmentalism"}
};

//=============================================================================
// Config Defaults
//=============================================================================

void style_representer_config_defaults(style_representer_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(style_representer_config_t));

    config->embedding_dim = DEFAULT_EMBEDDING_DIM;
    config->use_pretrained = false;

    config->load_literary_archetypes = true;
    config->load_music_archetypes = true;
    config->load_visual_archetypes = true;
    config->load_cinema_archetypes = true;

    config->enable_gpu = false;
    config->gpu_device_id = 0;
}

//=============================================================================
// Internal Helpers
//=============================================================================

static void generate_archetype_embedding(style_embedding_t* emb, uint32_t dim,
                                          uint32_t archetype_idx, uint32_t modality_offset) {
    /* Generate deterministic pseudo-random embedding based on archetype identity */
    uint32_t seed = modality_offset * 1000 + archetype_idx * 37;

    for (uint32_t i = 0; i < dim; i++) {
        /* LCG-based pseudo-random */
        seed = seed * 1103515245 + 12345;
        float val = ((float)(seed % 10000) / 5000.0f) - 1.0f;

        /* Add some structure based on position */
        float pos_bias = sinf((float)i * 0.1f + (float)archetype_idx) * 0.3f;
        emb->embedding[i] = val + pos_bias;
    }

    style_embedding_normalize(emb);
}

static int init_literary_archetypes(style_representer_t* repr) {
    repr->num_literary = STYLE_LIT_COUNT;
    repr->literary_archetypes = nimcp_calloc(repr->num_literary, sizeof(archetype_info_t));
    if (!repr->literary_archetypes) return -1;

    for (uint32_t i = 0; i < repr->num_literary; i++) {
        archetype_info_t* info = &repr->literary_archetypes[i];
        info->archetype_id = (int32_t)i;
        info->modality = ART_MODALITY_TEXT_POETRY;

        strncpy(info->name, literary_archetype_data[i].name, sizeof(info->name) - 1);
        strncpy(info->description, literary_archetype_data[i].description,
                sizeof(info->description) - 1);
        strncpy(info->era, literary_archetype_data[i].era, sizeof(info->era) - 1);
        strncpy(info->characteristics, literary_archetype_data[i].characteristics,
                sizeof(info->characteristics) - 1);

        style_embedding_create(&info->canonical, repr->embedding_dim);
        generate_archetype_embedding(&info->canonical, repr->embedding_dim, i, 0);
    }

    return 0;
}

static int init_music_archetypes(style_representer_t* repr) {
    repr->num_music = STYLE_MUSIC_COUNT;
    repr->music_archetypes = nimcp_calloc(repr->num_music, sizeof(archetype_info_t));
    if (!repr->music_archetypes) return -1;

    for (uint32_t i = 0; i < repr->num_music; i++) {
        archetype_info_t* info = &repr->music_archetypes[i];
        info->archetype_id = (int32_t)i;
        info->modality = ART_MODALITY_MUSIC_CLASSICAL;  /* Base music modality */

        strncpy(info->name, music_archetype_data[i].name, sizeof(info->name) - 1);
        strncpy(info->description, music_archetype_data[i].description,
                sizeof(info->description) - 1);
        strncpy(info->era, music_archetype_data[i].era, sizeof(info->era) - 1);
        strncpy(info->characteristics, music_archetype_data[i].characteristics,
                sizeof(info->characteristics) - 1);

        style_embedding_create(&info->canonical, repr->embedding_dim);
        generate_archetype_embedding(&info->canonical, repr->embedding_dim, i, 1);
    }

    return 0;
}

static int init_visual_archetypes(style_representer_t* repr) {
    repr->num_visual = STYLE_VIS_COUNT;
    repr->visual_archetypes = nimcp_calloc(repr->num_visual, sizeof(archetype_info_t));
    if (!repr->visual_archetypes) return -1;

    for (uint32_t i = 0; i < repr->num_visual; i++) {
        archetype_info_t* info = &repr->visual_archetypes[i];
        info->archetype_id = (int32_t)i;
        info->modality = ART_MODALITY_VISUAL_PAINTING;

        strncpy(info->name, visual_archetype_data[i].name, sizeof(info->name) - 1);
        strncpy(info->description, visual_archetype_data[i].description,
                sizeof(info->description) - 1);
        strncpy(info->era, visual_archetype_data[i].era, sizeof(info->era) - 1);
        strncpy(info->characteristics, visual_archetype_data[i].characteristics,
                sizeof(info->characteristics) - 1);

        style_embedding_create(&info->canonical, repr->embedding_dim);
        generate_archetype_embedding(&info->canonical, repr->embedding_dim, i, 2);
    }

    return 0;
}

static int init_cinema_archetypes(style_representer_t* repr) {
    repr->num_cinema = STYLE_CINEMA_COUNT;
    repr->cinema_archetypes = nimcp_calloc(repr->num_cinema, sizeof(archetype_info_t));
    if (!repr->cinema_archetypes) return -1;

    for (uint32_t i = 0; i < repr->num_cinema; i++) {
        archetype_info_t* info = &repr->cinema_archetypes[i];
        info->archetype_id = (int32_t)i;
        info->modality = ART_MODALITY_VIDEO_CINEMA;  /* Base cinema modality */

        strncpy(info->name, cinema_archetype_data[i].name, sizeof(info->name) - 1);
        strncpy(info->description, cinema_archetype_data[i].description,
                sizeof(info->description) - 1);
        strncpy(info->era, cinema_archetype_data[i].era, sizeof(info->era) - 1);
        strncpy(info->characteristics, cinema_archetype_data[i].characteristics,
                sizeof(info->characteristics) - 1);

        style_embedding_create(&info->canonical, repr->embedding_dim);
        generate_archetype_embedding(&info->canonical, repr->embedding_dim, i, 3);
    }

    return 0;
}

static void free_archetypes(archetype_info_t* archetypes, uint32_t count) {
    if (!archetypes) return;

    for (uint32_t i = 0; i < count; i++) {
        style_embedding_destroy(&archetypes[i].canonical);
    }
    nimcp_free(archetypes);
}

//=============================================================================
// Lifecycle API
//=============================================================================

style_representer_t* style_representer_create(
    const style_representer_config_t* config) {

    style_representer_t* repr = nimcp_calloc(1, sizeof(style_representer_t));
    if (!repr) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate style representer");
        return NULL;
    }

    if (config) {
        repr->config = *config;
    } else {
        style_representer_config_defaults(&repr->config);
    }

    repr->embedding_dim = repr->config.embedding_dim;

    /* Initialize archetypes */
    if (repr->config.load_literary_archetypes) {
        if (init_literary_archetypes(repr) < 0) {
            LOG_ERROR(LOG_MODULE, "Failed to init literary archetypes");
        }
    }

    if (repr->config.load_music_archetypes) {
        if (init_music_archetypes(repr) < 0) {
            LOG_ERROR(LOG_MODULE, "Failed to init music archetypes");
        }
    }

    if (repr->config.load_visual_archetypes) {
        if (init_visual_archetypes(repr) < 0) {
            LOG_ERROR(LOG_MODULE, "Failed to init visual archetypes");
        }
    }

    if (repr->config.load_cinema_archetypes) {
        if (init_cinema_archetypes(repr) < 0) {
            LOG_ERROR(LOG_MODULE, "Failed to init cinema archetypes");
        }
    }

    LOG_INFO(LOG_MODULE, "Style representer created (dim=%u, lit=%u, mus=%u, vis=%u, cin=%u)",
             repr->embedding_dim, repr->num_literary, repr->num_music,
             repr->num_visual, repr->num_cinema);

    return repr;
}

void style_representer_destroy(style_representer_t* repr) {
    if (!repr) return;

    free_archetypes(repr->literary_archetypes, repr->num_literary);
    free_archetypes(repr->music_archetypes, repr->num_music);
    free_archetypes(repr->visual_archetypes, repr->num_visual);
    free_archetypes(repr->cinema_archetypes, repr->num_cinema);

    nimcp_free(repr);

    LOG_INFO(LOG_MODULE, "Style representer destroyed");
}

//=============================================================================
// Embedding API
//=============================================================================

int style_repr_embed_text(style_representer_t* repr,
                          const char* text, size_t len,
                          art_modality_t modality,
                          style_embedding_t* out) {
    if (!repr || !text || !out) return -1;
    (void)modality;

    style_embedding_create(out, repr->embedding_dim);

    /* Extract text features and create embedding */
    /* Simple feature extraction based on text statistics */

    uint32_t word_count = 0;
    uint32_t char_count = 0;
    uint32_t sentence_count = 0;
    uint8_t char_freq[26] = {0};

    for (size_t i = 0; i < len; i++) {
        char c = text[i];
        char_count++;

        if (c == ' ' || c == '\n') {
            word_count++;
        } else if (c == '.' || c == '!' || c == '?') {
            sentence_count++;
        } else if (c >= 'a' && c <= 'z') {
            char_freq[c - 'a']++;
        } else if (c >= 'A' && c <= 'Z') {
            char_freq[c - 'A']++;
        }
    }
    word_count++;  /* Last word */

    /* Generate embedding from features */
    float avg_word_len = word_count > 0 ? (float)char_count / word_count : 0.0f;
    float avg_sent_len = sentence_count > 0 ? (float)word_count / sentence_count : 0.0f;

    for (uint32_t i = 0; i < repr->embedding_dim; i++) {
        float val = 0.0f;

        /* Feature-based components */
        val += sinf((float)i * avg_word_len * 0.1f) * 0.3f;
        val += cosf((float)i * avg_sent_len * 0.05f) * 0.3f;

        /* Character frequency component */
        uint32_t char_idx = i % 26;
        float freq = char_count > 0 ? (float)char_freq[char_idx] / char_count : 0.0f;
        val += freq * 2.0f - 0.1f;

        /* Position-based bias */
        val += sinf((float)i * 0.05f) * 0.2f;

        out->embedding[i] = val;
    }

    style_embedding_normalize(out);
    repr->embeddings_computed++;

    return 0;
}

int style_repr_embed_music(style_representer_t* repr,
                           const music_track_t* tracks, uint32_t num_tracks,
                           style_embedding_t* out) {
    if (!repr || !tracks || !out) return -1;

    style_embedding_create(out, repr->embedding_dim);

    /* Aggregate features from all tracks */
    float avg_pitch = 0.0f;
    float avg_velocity = 0.0f;
    float avg_duration = 0.0f;
    uint32_t total_notes = 0;

    for (uint32_t t = 0; t < num_tracks; t++) {
        for (uint32_t n = 0; n < tracks[t].num_notes; n++) {
            avg_pitch += tracks[t].notes[n].pitch;
            avg_velocity += tracks[t].notes[n].velocity;
            avg_duration += tracks[t].notes[n].duration_beats;
            total_notes++;
        }
    }

    if (total_notes > 0) {
        avg_pitch /= total_notes;
        avg_velocity /= total_notes;
        avg_duration /= total_notes;
    }

    /* Generate embedding from music features */
    for (uint32_t i = 0; i < repr->embedding_dim; i++) {
        float val = 0.0f;

        val += sinf((float)i * avg_pitch * 0.01f) * 0.4f;
        val += cosf((float)i * avg_velocity * 0.5f) * 0.3f;
        val += sinf((float)i * avg_duration * 0.2f) * 0.3f;

        out->embedding[i] = val;
    }

    style_embedding_normalize(out);
    repr->embeddings_computed++;

    return 0;
}

int style_repr_embed_visual(style_representer_t* repr,
                            const visual_image_t* image,
                            style_embedding_t* out) {
    if (!repr || !image || !out) return -1;

    style_embedding_create(out, repr->embedding_dim);

    /* Extract visual features */
    float avg_r = 0.0f, avg_g = 0.0f, avg_b = 0.0f;
    uint64_t pixels = (uint64_t)image->width * image->height;

    if (image->pixels && pixels > 0) {
        /* Sample pixels for average color */
        uint64_t sample_step = pixels > 1000 ? pixels / 1000 : 1;
        uint32_t samples = 0;

        for (uint64_t i = 0; i < pixels; i += sample_step) {
            uint32_t idx = (uint32_t)(i * image->channels);
            if (image->channels >= 3) {
                avg_r += image->pixels[idx];
                avg_g += image->pixels[idx + 1];
                avg_b += image->pixels[idx + 2];
            } else {
                avg_r += image->pixels[idx];
                avg_g += image->pixels[idx];
                avg_b += image->pixels[idx];
            }
            samples++;
        }

        if (samples > 0) {
            avg_r /= samples * 255.0f;
            avg_g /= samples * 255.0f;
            avg_b /= samples * 255.0f;
        }
    }

    /* Generate embedding from visual features */
    float aspect = image->height > 0 ? (float)image->width / image->height : 1.0f;

    for (uint32_t i = 0; i < repr->embedding_dim; i++) {
        float val = 0.0f;

        val += sinf((float)i * avg_r * 3.14159f) * 0.3f;
        val += cosf((float)i * avg_g * 3.14159f) * 0.3f;
        val += sinf((float)i * avg_b * 3.14159f + 1.0f) * 0.2f;
        val += cosf((float)i * aspect * 0.1f) * 0.2f;

        out->embedding[i] = val;
    }

    style_embedding_normalize(out);
    repr->embeddings_computed++;

    return 0;
}

int style_repr_embed_audio(style_representer_t* repr,
                           const float* audio, uint64_t num_samples,
                           uint32_t sample_rate,
                           style_embedding_t* out) {
    if (!repr || !audio || !out) return -1;

    style_embedding_create(out, repr->embedding_dim);

    /* Compute simple audio features */
    float rms = 0.0f;
    float zcr = 0.0f;  /* Zero crossing rate */

    for (uint64_t i = 0; i < num_samples; i++) {
        rms += audio[i] * audio[i];
        if (i > 0 && ((audio[i] >= 0) != (audio[i - 1] >= 0))) {
            zcr += 1.0f;
        }
    }

    rms = sqrtf(rms / num_samples);
    zcr = zcr / num_samples;

    float duration = (float)num_samples / sample_rate;

    /* Generate embedding from audio features */
    for (uint32_t i = 0; i < repr->embedding_dim; i++) {
        float val = 0.0f;

        val += sinf((float)i * rms * 10.0f) * 0.4f;
        val += cosf((float)i * zcr * 100.0f) * 0.3f;
        val += sinf((float)i * duration * 0.01f) * 0.3f;

        out->embedding[i] = val;
    }

    style_embedding_normalize(out);
    repr->embeddings_computed++;

    return 0;
}

//=============================================================================
// Archetype API
//=============================================================================

int style_repr_get_archetype_info(const style_representer_t* repr,
                                   art_modality_t modality,
                                   int32_t archetype_id,
                                   archetype_info_t* out) {
    if (!repr || !out || archetype_id < 0) return -1;

    archetype_info_t* archetypes = NULL;
    uint32_t count = 0;

    if (art_modality_is_text(modality)) {
        archetypes = repr->literary_archetypes;
        count = repr->num_literary;
    } else if (art_modality_category(modality) == 1) {
        archetypes = repr->music_archetypes;
        count = repr->num_music;
    } else if (art_modality_category(modality) == 2) {  /* Visual */
        archetypes = repr->visual_archetypes;
        count = repr->num_visual;
    } else if (art_modality_category(modality) == 3) {  /* Video/Film */
        archetypes = repr->cinema_archetypes;
        count = repr->num_cinema;
    }

    if (!archetypes || (uint32_t)archetype_id >= count) return -1;

    /* Copy info (shallow copy of embedding) */
    *out = archetypes[archetype_id];

    return 0;
}

int style_repr_get_archetype_embedding(const style_representer_t* repr,
                                        art_modality_t modality,
                                        int32_t archetype_id,
                                        style_embedding_t* out) {
    if (!repr || !out || archetype_id < 0) return -1;

    archetype_info_t* archetypes = NULL;
    uint32_t count = 0;

    if (art_modality_is_text(modality)) {
        archetypes = repr->literary_archetypes;
        count = repr->num_literary;
    } else if (art_modality_category(modality) == 1) {
        archetypes = repr->music_archetypes;
        count = repr->num_music;
    } else if (art_modality_category(modality) == 2) {  /* Visual */
        archetypes = repr->visual_archetypes;
        count = repr->num_visual;
    } else if (art_modality_category(modality) == 3) {  /* Video/Film */
        archetypes = repr->cinema_archetypes;
        count = repr->num_cinema;
    }

    if (!archetypes || (uint32_t)archetype_id >= count) return -1;

    return style_embedding_clone(&archetypes[archetype_id].canonical, out);
}

uint32_t style_repr_list_archetypes(const style_representer_t* repr,
                                     art_modality_t modality,
                                     archetype_info_t* out,
                                     uint32_t max_count) {
    if (!repr || !out) return 0;

    archetype_info_t* archetypes = NULL;
    uint32_t count = 0;

    if (art_modality_is_text(modality)) {
        archetypes = repr->literary_archetypes;
        count = repr->num_literary;
    } else if (art_modality_category(modality) == 1) {
        archetypes = repr->music_archetypes;
        count = repr->num_music;
    } else if (art_modality_category(modality) == 2) {  /* Visual */
        archetypes = repr->visual_archetypes;
        count = repr->num_visual;
    } else if (art_modality_category(modality) == 3) {  /* Video/Film */
        archetypes = repr->cinema_archetypes;
        count = repr->num_cinema;
    }

    if (!archetypes) return 0;

    uint32_t copy_count = count < max_count ? count : max_count;
    memcpy(out, archetypes, copy_count * sizeof(archetype_info_t));

    return copy_count;
}

int style_repr_find_archetype_by_name(const style_representer_t* repr,
                                       art_modality_t modality,
                                       const char* name,
                                       archetype_info_t* out) {
    if (!repr || !name || !out) return -1;

    archetype_info_t* archetypes = NULL;
    uint32_t count = 0;

    if (art_modality_is_text(modality)) {
        archetypes = repr->literary_archetypes;
        count = repr->num_literary;
    } else if (art_modality_category(modality) == 1) {
        archetypes = repr->music_archetypes;
        count = repr->num_music;
    } else if (art_modality_category(modality) == 2) {  /* Visual */
        archetypes = repr->visual_archetypes;
        count = repr->num_visual;
    } else if (art_modality_category(modality) == 3) {  /* Video/Film */
        archetypes = repr->cinema_archetypes;
        count = repr->num_cinema;
    }

    if (!archetypes) return -1;

    for (uint32_t i = 0; i < count; i++) {
        if (strcasecmp(archetypes[i].name, name) == 0) {
            *out = archetypes[i];
            return 0;
        }
    }

    return -1;
}

//=============================================================================
// Operations API
//=============================================================================

float style_repr_similarity(const style_embedding_t* a,
                            const style_embedding_t* b) {
    return style_embedding_similarity(a, b);
}

float style_repr_distance(const style_embedding_t* a,
                          const style_embedding_t* b) {
    if (!a || !b || a->embedding_dim != b->embedding_dim) return -1.0f;

    float sum = 0.0f;
    for (uint32_t i = 0; i < a->embedding_dim; i++) {
        float diff = a->embedding[i] - b->embedding[i];
        sum += diff * diff;
    }

    return sqrtf(sum);
}

int style_repr_interpolate(const style_embedding_t* a,
                           const style_embedding_t* b,
                           float t,
                           style_embedding_t* out) {
    if (!a || !b || !out || a->embedding_dim != b->embedding_dim) return -1;

    style_embedding_create(out, a->embedding_dim);

    for (uint32_t i = 0; i < a->embedding_dim; i++) {
        out->embedding[i] = a->embedding[i] * (1.0f - t) + b->embedding[i] * t;
    }

    return 0;
}

int style_repr_combine(const style_embedding_t* a,
                       const style_embedding_t* b,
                       float scale_a, float scale_b,
                       style_embedding_t* out) {
    if (!a || !b || !out || a->embedding_dim != b->embedding_dim) return -1;

    style_embedding_create(out, a->embedding_dim);

    for (uint32_t i = 0; i < a->embedding_dim; i++) {
        out->embedding[i] = a->embedding[i] * scale_a + b->embedding[i] * scale_b;
    }

    style_embedding_normalize(out);

    return 0;
}

int style_repr_negate(const style_embedding_t* style,
                      style_embedding_t* out) {
    if (!style || !out) return -1;

    style_embedding_create(out, style->embedding_dim);

    for (uint32_t i = 0; i < style->embedding_dim; i++) {
        out->embedding[i] = -style->embedding[i];
    }

    return 0;
}

int32_t style_repr_project_to_archetype(const style_representer_t* repr,
                                         const style_embedding_t* style,
                                         style_embedding_t* out) {
    if (!repr || !style || !out) return -1;

    /* Find closest archetype across all modalities */
    float best_sim = -2.0f;
    int32_t best_id = -1;
    archetype_info_t* best_archetype = NULL;

    /* Check literary */
    for (uint32_t i = 0; i < repr->num_literary; i++) {
        float sim = style_embedding_similarity(style, &repr->literary_archetypes[i].canonical);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = (int32_t)i;
            best_archetype = &repr->literary_archetypes[i];
        }
    }

    /* Check music */
    for (uint32_t i = 0; i < repr->num_music; i++) {
        float sim = style_embedding_similarity(style, &repr->music_archetypes[i].canonical);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = (int32_t)(i + 100);  /* Offset for music */
            best_archetype = &repr->music_archetypes[i];
        }
    }

    /* Check visual */
    for (uint32_t i = 0; i < repr->num_visual; i++) {
        float sim = style_embedding_similarity(style, &repr->visual_archetypes[i].canonical);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = (int32_t)(i + 200);  /* Offset for visual */
            best_archetype = &repr->visual_archetypes[i];
        }
    }

    /* Check cinema */
    for (uint32_t i = 0; i < repr->num_cinema; i++) {
        float sim = style_embedding_similarity(style, &repr->cinema_archetypes[i].canonical);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = (int32_t)(i + 300);  /* Offset for cinema */
            best_archetype = &repr->cinema_archetypes[i];
        }
    }

    if (best_archetype) {
        style_embedding_clone(&best_archetype->canonical, out);
    }

    return best_id;
}

//=============================================================================
// Cortical Integration API
//=============================================================================

void style_representer_set_cortical_columns(style_representer_t* repr,
                                             void* cortical_columns) {
    if (!repr) return;
    repr->cortical_columns = cortical_columns;
}

//=============================================================================
// Utility API
//=============================================================================

style_embedding_t* style_repr_alloc_embedding(uint32_t dim) {
    style_embedding_t* emb = nimcp_calloc(1, sizeof(style_embedding_t));
    if (!emb) return NULL;

    if (style_embedding_create(emb, dim) < 0) {
        nimcp_free(emb);
        return NULL;
    }

    return emb;
}

style_embedding_t* style_repr_clone_embedding(const style_embedding_t* src) {
    if (!src) return NULL;

    style_embedding_t* clone = nimcp_calloc(1, sizeof(style_embedding_t));
    if (!clone) return NULL;

    if (style_embedding_clone(src, clone) < 0) {
        nimcp_free(clone);
        return NULL;
    }

    return clone;
}

void style_repr_normalize(style_embedding_t* style) {
    style_embedding_normalize(style);
}

uint32_t style_repr_get_dim(const style_representer_t* repr) {
    return repr ? repr->embedding_dim : 0;
}
