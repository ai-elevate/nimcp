/**
 * @file nimcp_grounded_language.c
 * @brief Grounded Language System - Implementation
 *
 * Human-like language through grounded semantics, not token statistics.
 * Words are learned by binding to multimodal concept representations.
 * Production works by navigating semantic space and finding words.
 *
 * @version 1.0.0
 * @date 2026-03-07
 */

#include "language/nimcp_grounded_language.h"
#include "cognitive/memory/nimcp_semantic_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define LOG_MODULE "GROUNDED_LANG"

/*=============================================================================
 * Internal Structures
 *===========================================================================*/

/**
 * @brief Context buffer for tracking conversation state
 */
typedef struct {
    float*   context_vector;          /**< Running context [semantic_dim] */
    float    context_strength;        /**< How strong current context is */
    uint32_t words_in_context;        /**< Words seen in current context */
    uint64_t last_concepts[16];       /**< Recent concept activations */
    uint32_t last_concept_count;      /**< Number of recent concepts */
} gl_context_t;

/**
 * @brief Internal state of the grounded language system
 */
struct grounded_language {
    /* Lexicon: word forms -> concept bindings */
    gl_lexicon_entry_t** lexicon;       /**< Hash table of lexicon entries */
    uint32_t             lexicon_size;  /**< Hash table size */
    uint32_t             vocab_count;   /**< Number of words */
    gl_lexicon_entry_t** vocab_list;    /**< Linear list for iteration */

    /* Syntactic templates */
    gl_template_t*       templates;     /**< Array of syntactic templates */
    uint32_t             template_count;
    uint32_t             template_capacity;

    /* Semantic integration */
    uint32_t             semantic_dim;  /**< Dimension of semantic vectors */
    semantic_memory_system_t* semantic_memory; /**< Brain's concept store */

    /* Context state */
    gl_context_t         context;       /**< Current conversation context */

    /* Cross-modal connections */
    void*                visual_ctx;
    void*                auditory_ctx;
    void*                speech_ctx;
    void*                column_pool;
    void*                emotional_ctx;

    /* Learning parameters */
    float                hebbian_lr;    /**< Hebbian learning rate */
    float                decay_rate;    /**< Association decay rate */

    /* Statistics */
    gl_stats_t           stats;

    /* RNG state for sampling */
    uint64_t             rng_state;
};

/*=============================================================================
 * Utility Functions
 *===========================================================================*/

/** Simple string hash (FNV-1a) */
static uint32_t hash_word(const char* word) {
    uint32_t hash = 2166136261u;
    for (const char* p = word; *p; p++) {
        hash ^= (uint32_t)(unsigned char)tolower(*p);
        hash *= 16777619u;
    }
    return hash;
}

/** Lowercase a word in-place into buffer */
static void lowercase_word(const char* src, char* dst, size_t max_len) {
    size_t i;
    for (i = 0; i < max_len - 1 && src[i]; i++) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

/** Cosine similarity between two vectors */
static float cosine_similarity(const float* a, const float* b, uint32_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-8f) return 0.0f;
    return dot / denom;
}

/** Simple xorshift64 RNG */
static float gl_random(grounded_language_t* gl) {
    gl->rng_state ^= gl->rng_state << 13;
    gl->rng_state ^= gl->rng_state >> 7;
    gl->rng_state ^= gl->rng_state << 17;
    return (float)(gl->rng_state & 0xFFFFFF) / (float)0xFFFFFF;
}

/** Normalize a vector in-place */
static void normalize_vector(float* vec, uint32_t dim) {
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-8f) return;
    float inv = 1.0f / norm;
    for (uint32_t i = 0; i < dim; i++) {
        vec[i] *= inv;
    }
}

/*=============================================================================
 * Lexicon Operations
 *===========================================================================*/

/** Find or create a lexicon entry for a word */
static gl_lexicon_entry_t* lexicon_find_or_create(grounded_language_t* gl, const char* word) {
    char lower[GL_MAX_WORD_LEN];
    lowercase_word(word, lower, GL_MAX_WORD_LEN);
    uint32_t h = hash_word(lower);
    uint32_t idx = h % gl->lexicon_size;

    /* Linear probe to find existing entry */
    for (uint32_t probe = 0; probe < gl->lexicon_size; probe++) {
        uint32_t slot = (idx + probe) % gl->lexicon_size;
        if (!gl->lexicon[slot]) {
            /* Empty slot - create new entry */
            if (gl->vocab_count >= GL_MAX_VOCAB) {
                LOG_WARN(LOG_MODULE, "Lexicon full (%u words)", gl->vocab_count);
                return NULL;
            }

            gl_lexicon_entry_t* entry = (gl_lexicon_entry_t*)nimcp_calloc(1, sizeof(gl_lexicon_entry_t));
            if (!entry) return NULL;

            strncpy(entry->form, lower, GL_MAX_WORD_LEN - 1);
            entry->form_hash = h;
            entry->binding_capacity = 4; /* Start small, grow */
            entry->bindings = (gl_word_binding_t*)nimcp_calloc(entry->binding_capacity,
                                                                sizeof(gl_word_binding_t));
            if (!entry->bindings) {
                nimcp_free(entry);
                return NULL;
            }

            entry->context_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
            if (!entry->context_vector) {
                nimcp_free(entry->bindings);
                nimcp_free(entry);
                return NULL;
            }

            gl->lexicon[slot] = entry;
            gl->vocab_list[gl->vocab_count] = entry;
            gl->vocab_count++;
            gl->stats.vocab_size = gl->vocab_count;
            return entry;
        }

        if (gl->lexicon[slot]->form_hash == h &&
            strcmp(gl->lexicon[slot]->form, lower) == 0) {
            return gl->lexicon[slot]; /* Found existing */
        }
    }

    return NULL; /* Table full (shouldn't happen with GL_MAX_VOCAB < table size) */
}

/** Find a lexicon entry (read-only, no create) */
static const gl_lexicon_entry_t* lexicon_find(const grounded_language_t* gl, const char* word) {
    char lower[GL_MAX_WORD_LEN];
    lowercase_word(word, lower, GL_MAX_WORD_LEN);
    uint32_t h = hash_word(lower);
    uint32_t idx = h % gl->lexicon_size;

    for (uint32_t probe = 0; probe < gl->lexicon_size; probe++) {
        uint32_t slot = (idx + probe) % gl->lexicon_size;
        if (!gl->lexicon[slot]) return NULL;
        if (gl->lexicon[slot]->form_hash == h &&
            strcmp(gl->lexicon[slot]->form, lower) == 0) {
            return gl->lexicon[slot];
        }
    }
    return NULL;
}

/** Add or strengthen a binding between a word and a concept */
static int lexicon_bind(grounded_language_t* gl, gl_lexicon_entry_t* entry,
                        uint64_t concept_id, float strength, gl_modality_t modality) {
    /* Check if binding already exists */
    for (uint32_t i = 0; i < entry->binding_count; i++) {
        if (entry->bindings[i].concept_id == concept_id) {
            /* Hebbian strengthening: delta = lr * (1 - current) * input_strength */
            float delta = gl->hebbian_lr * (1.0f - entry->bindings[i].strength) * strength;
            entry->bindings[i].strength += delta;
            if (entry->bindings[i].strength > 1.0f)
                entry->bindings[i].strength = 1.0f;

            /* Strengthen specific modality */
            entry->bindings[i].modality_strength[modality] += gl->hebbian_lr * strength;
            if (entry->bindings[i].modality_strength[modality] > 1.0f)
                entry->bindings[i].modality_strength[modality] = 1.0f;

            entry->bindings[i].exposure_count++;
            entry->bindings[i].confidence = 1.0f - expf(-(float)entry->bindings[i].exposure_count / 5.0f);
            entry->bindings[i].last_activation_ms = (uint64_t)time(NULL) * 1000;
            gl->stats.total_bindings++;
            return 0;
        }
    }

    /* New binding - grow array if needed */
    if (entry->binding_count >= entry->binding_capacity) {
        uint32_t new_cap = entry->binding_capacity * 2;
        gl_word_binding_t* new_bindings = (gl_word_binding_t*)nimcp_realloc(
            entry->bindings, new_cap * sizeof(gl_word_binding_t));
        if (!new_bindings) return -1;
        memset(new_bindings + entry->binding_capacity, 0,
               (new_cap - entry->binding_capacity) * sizeof(gl_word_binding_t));
        entry->bindings = new_bindings;
        entry->binding_capacity = new_cap;
    }

    /* Create new binding */
    gl_word_binding_t* b = &entry->bindings[entry->binding_count];
    b->concept_id = concept_id;
    b->strength = strength;
    b->modality_strength[modality] = strength;
    b->exposure_count = 1;
    b->confidence = 0.2f; /* Low initial confidence */
    b->last_activation_ms = (uint64_t)time(NULL) * 1000;

    entry->binding_count++;
    gl->stats.total_bindings++;
    return 0;
}

/*=============================================================================
 * Tokenization (simple whitespace + punctuation)
 *===========================================================================*/

/** Split text into words. Returns word count. words[] points into buf (modified). */
static uint32_t tokenize_text(char* buf, char** words, uint32_t max_words) {
    uint32_t count = 0;
    char* p = buf;

    while (*p && count < max_words) {
        /* Skip whitespace and punctuation (but keep track of punctuation) */
        while (*p && (isspace((unsigned char)*p) || ispunct((unsigned char)*p))) {
            /* If it's meaningful punctuation, emit as separate token */
            if (*p == '.' || *p == '?' || *p == '!' || *p == ',') {
                /* Store the punctuation char at position, null-terminate it */
                char punct = *p;
                *p = '\0';
                p++;
                /* Skip if we can't store it */
                if (count < max_words) {
                    /* We'll skip punctuation tokens for now - they don't map to concepts */
                }
                (void)punct;
            } else {
                p++;
            }
        }
        if (!*p) break;

        /* Start of word */
        words[count] = p;

        /* Find end of word */
        while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p)) {
            p++;
        }

        if (*p) {
            *p = '\0';
            p++;
        }
        count++;
    }

    return count;
}

/*=============================================================================
 * Built-in Function Words and Templates
 *===========================================================================*/

/** Basic English function words that provide syntactic scaffolding */
static const char* FUNCTION_WORDS[] = {
    "the", "a", "an", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "in", "on", "at", "to", "for", "with", "by", "from",
    "of", "and", "or", "but", "not", "no", "if", "then",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "what", "who", "how", "when", "where", "why",
    "very", "more", "most", "also", "just", "still",
    NULL
};

/** Seed the lexicon with function words */
static void seed_function_words(grounded_language_t* gl) {
    for (int i = 0; FUNCTION_WORDS[i]; i++) {
        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, FUNCTION_WORDS[i]);
        if (entry) {
            entry->learned_class = GL_CLASS_FUNCTION;
            entry->class_confidence = 0.9f;
            entry->frequency = 100; /* High prior frequency */
        }
    }
    /* Mark pronouns specifically */
    const char* pronouns[] = {"i", "you", "he", "she", "it", "we", "they", NULL};
    for (int i = 0; pronouns[i]; i++) {
        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, pronouns[i]);
        if (entry) entry->learned_class = GL_CLASS_PRONOUN;
    }
}

/** Create built-in syntactic templates */
static void seed_templates(grounded_language_t* gl) {
    /* SVO: "The dog chases the cat" */
    gl_template_t* t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_SVO;
    t->slots[0] = GL_CLASS_NOUN;
    t->slots[1] = GL_CLASS_VERB;
    t->slots[2] = GL_CLASS_NOUN;
    t->slot_count = 3;
    t->frequency = 10.0f;
    t->confidence = 0.8f;

    /* SV: "The bird flies" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_SV;
    t->slots[0] = GL_CLASS_NOUN;
    t->slots[1] = GL_CLASS_VERB;
    t->slot_count = 2;
    t->frequency = 8.0f;
    t->confidence = 0.8f;

    /* Copula: "The sky is blue" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_COPULA;
    t->slots[0] = GL_CLASS_NOUN;
    t->slots[1] = GL_CLASS_FUNCTION; /* is/are */
    t->slots[2] = GL_CLASS_ADJECTIVE;
    t->slot_count = 3;
    t->frequency = 8.0f;
    t->confidence = 0.8f;

    /* NP: "the big red dog" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_NP;
    t->slots[0] = GL_CLASS_ADJECTIVE;
    t->slots[1] = GL_CLASS_NOUN;
    t->slot_count = 2;
    t->frequency = 6.0f;
    t->confidence = 0.7f;

    /* SVA: "The cat sits quietly" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_SVA;
    t->slots[0] = GL_CLASS_NOUN;
    t->slots[1] = GL_CLASS_VERB;
    t->slots[2] = GL_CLASS_ADVERB;
    t->slot_count = 3;
    t->frequency = 5.0f;
    t->confidence = 0.7f;
}

/*=============================================================================
 * Concept Operations (via Semantic Memory)
 *===========================================================================*/

/** Find or create a concept from features */
static uint64_t find_or_create_concept(grounded_language_t* gl,
                                        const float* features, uint32_t dim,
                                        const char* label, uint32_t category) {
    if (!gl->semantic_memory) {
        /* No semantic memory — create a pseudo-concept ID from feature hash */
        uint32_t hash = 0;
        for (uint32_t i = 0; i < dim && i < 8; i++) {
            uint32_t bits;
            memcpy(&bits, &features[i], sizeof(bits));
            hash ^= bits;
        }
        return (uint64_t)hash | 0x100000000ULL; /* High bit marks pseudo-concept */
    }

    /* Search for similar existing concept */
    /* Use semantic_memory_find_similar if available */
    semantic_memory_system_t* sm = gl->semantic_memory;

    /* Linear scan for now — find closest concept */
    float best_sim = -1.0f;
    uint64_t best_id = 0;

    for (uint32_t i = 0; i < sm->concept_count; i++) {
        if (!sm->concepts[i]) continue;
        const semantic_concept_t* c = sm->concepts[i];
        if (c->feature_dim != dim && c->feature_dim != gl->semantic_dim) continue;

        uint32_t cmp_dim = (dim < c->feature_dim) ? dim : c->feature_dim;
        float sim = cosine_similarity(features, c->features, cmp_dim);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = c->id;
        }
    }

    /* If very similar concept exists, return it */
    if (best_sim > 0.85f && best_id != 0) {
        return best_id;
    }

    /* Create new concept */
    /* Pad or truncate features to semantic_dim */
    float* padded = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!padded) return 0;
    uint32_t copy_dim = (dim < gl->semantic_dim) ? dim : gl->semantic_dim;
    memcpy(padded, features, copy_dim * sizeof(float));

    uint64_t id = semantic_memory_create_concept(
        sm, padded, gl->semantic_dim, label, (concept_category_t)category);

    nimcp_free(padded);
    return id;
}

/** Get concept features (returns pointer, do not free) */
static const float* get_concept_features(const grounded_language_t* gl, uint64_t concept_id) {
    if (!gl->semantic_memory) return NULL;
    const semantic_concept_t* c = semantic_memory_get_concept(gl->semantic_memory, concept_id);
    if (!c) return NULL;
    return c->features;
}

/*=============================================================================
 * Word Selection for Production
 *===========================================================================*/

/** Find the best word for a concept (highest binding strength) */
static const char* best_word_for_concept(const grounded_language_t* gl, uint64_t concept_id) {
    const char* best = NULL;
    float best_strength = -1.0f;

    for (uint32_t w = 0; w < gl->vocab_count; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry) continue;
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            if (entry->bindings[b].concept_id == concept_id &&
                entry->bindings[b].strength > best_strength) {
                best_strength = entry->bindings[b].strength;
                best = entry->form;
            }
        }
    }
    return best;
}

/** Find words closest to a semantic vector */
static uint32_t find_words_near_vector(const grounded_language_t* gl,
                                        const float* target, uint32_t dim,
                                        gl_word_class_t required_class,
                                        const char** out_words, float* out_scores,
                                        uint32_t max_words) {
    /* Score each word by how close its context vector is to target */
    uint32_t count = 0;

    for (uint32_t w = 0; w < gl->vocab_count && count < max_words; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry || !entry->context_initialized) continue;

        /* Filter by word class if specified */
        if (required_class != GL_CLASS_UNKNOWN && entry->learned_class != required_class) {
            continue;
        }

        uint32_t cmp_dim = (dim < gl->semantic_dim) ? dim : gl->semantic_dim;
        float sim = cosine_similarity(target, entry->context_vector, cmp_dim);

        /* Also check concept bindings — words bound to concepts near target */
        float concept_sim = 0.0f;
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            const float* cf = get_concept_features(gl, entry->bindings[b].concept_id);
            if (cf) {
                float cs = cosine_similarity(target, cf, cmp_dim);
                cs *= entry->bindings[b].strength; /* Weight by binding strength */
                if (cs > concept_sim) concept_sim = cs;
            }
        }

        /* Combined score: distributional + grounded */
        float score = 0.4f * sim + 0.6f * concept_sim;

        /* Insert into sorted output (simple insertion sort) */
        uint32_t insert_pos = count;
        for (uint32_t j = 0; j < count; j++) {
            if (score > out_scores[j]) {
                insert_pos = j;
                break;
            }
        }

        if (insert_pos < max_words) {
            /* Shift down */
            if (count < max_words) count++;
            for (uint32_t j = count - 1; j > insert_pos; j--) {
                out_words[j] = out_words[j - 1];
                out_scores[j] = out_scores[j - 1];
            }
            out_words[insert_pos] = entry->form;
            out_scores[insert_pos] = score;
        }
    }

    return count;
}

/*=============================================================================
 * Template-based Production
 *===========================================================================*/

/** Select best template for a given set of word classes */
static const gl_template_t* select_template(const grounded_language_t* gl,
                                             bool has_noun, bool has_verb,
                                             bool has_adj) {
    float best_score = -1.0f;
    const gl_template_t* best = NULL;

    for (uint32_t i = 0; i < gl->template_count; i++) {
        const gl_template_t* t = &gl->templates[i];
        float score = t->frequency * t->confidence;

        /* Bonus for templates that match available word classes */
        bool can_fill = true;
        for (uint32_t s = 0; s < t->slot_count; s++) {
            switch (t->slots[s]) {
                case GL_CLASS_NOUN: if (!has_noun) can_fill = false; break;
                case GL_CLASS_VERB: if (!has_verb) can_fill = false; break;
                case GL_CLASS_ADJECTIVE: if (!has_adj) can_fill = false; break;
                default: break; /* Function words always available */
            }
        }
        if (!can_fill) continue;

        if (score > best_score) {
            best_score = score;
            best = t;
        }
    }

    return best;
}

/** Get a copula form for simple sentences */
static const char* get_copula(void) {
    return "is";
}

/** Get a determiner */
static const char* get_determiner(grounded_language_t* gl) {
    return (gl_random(gl) > 0.5f) ? "the" : "a";
}

/*=============================================================================
 * Lifecycle Implementation
 *===========================================================================*/

grounded_language_t* grounded_language_create(uint32_t semantic_dim, void* semantic_memory) {
    grounded_language_t* gl = (grounded_language_t*)nimcp_calloc(1, sizeof(grounded_language_t));
    if (!gl) return NULL;

    gl->semantic_dim = (semantic_dim > 0) ? semantic_dim : GL_SEMANTIC_DIM;
    gl->semantic_memory = (semantic_memory_system_t*)semantic_memory;

    /* Hash table: 2x max vocab for low collision rate */
    gl->lexicon_size = GL_MAX_VOCAB * 2;
    gl->lexicon = (gl_lexicon_entry_t**)nimcp_calloc(gl->lexicon_size, sizeof(gl_lexicon_entry_t*));
    gl->vocab_list = (gl_lexicon_entry_t**)nimcp_calloc(GL_MAX_VOCAB, sizeof(gl_lexicon_entry_t*));
    if (!gl->lexicon || !gl->vocab_list) {
        grounded_language_destroy(gl);
        return NULL;
    }

    /* Templates */
    gl->template_capacity = GL_MAX_TEMPLATES;
    gl->templates = (gl_template_t*)nimcp_calloc(gl->template_capacity, sizeof(gl_template_t));
    if (!gl->templates) {
        grounded_language_destroy(gl);
        return NULL;
    }

    /* Context */
    gl->context.context_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!gl->context.context_vector) {
        grounded_language_destroy(gl);
        return NULL;
    }

    /* Learning parameters */
    gl->hebbian_lr = GL_HEBBIAN_LR_DEFAULT;
    gl->decay_rate = GL_DECAY_RATE_DEFAULT;

    /* RNG */
    gl->rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEULL;

    /* Seed with function words and basic templates */
    seed_function_words(gl);
    seed_templates(gl);

    LOG_INFO(LOG_MODULE, "Grounded language system created (dim=%u, vocab=%u function words)",
             gl->semantic_dim, gl->vocab_count);

    return gl;
}

void grounded_language_destroy(grounded_language_t* gl) {
    if (!gl) return;

    /* Free lexicon entries */
    if (gl->vocab_list) {
        for (uint32_t i = 0; i < gl->vocab_count; i++) {
            if (gl->vocab_list[i]) {
                nimcp_free(gl->vocab_list[i]->bindings);
                nimcp_free(gl->vocab_list[i]->context_vector);
                nimcp_free(gl->vocab_list[i]);
            }
        }
    }

    nimcp_free(gl->lexicon);
    nimcp_free(gl->vocab_list);
    nimcp_free(gl->templates);
    nimcp_free(gl->context.context_vector);
    nimcp_free(gl);
}

void grounded_language_set_semantic_memory(grounded_language_t* gl, void* semantic_memory) {
    if (gl) gl->semantic_memory = (semantic_memory_system_t*)semantic_memory;
}

/*=============================================================================
 * Comprehension
 *===========================================================================*/

int grounded_language_comprehend(grounded_language_t* gl, const char* text,
                                  gl_comprehension_result_t* result) {
    if (!gl || !text || !result) return -1;
    memset(result, 0, sizeof(*result));

    /* Tokenize */
    size_t text_len = strlen(text);
    char* buf = (char*)nimcp_malloc(text_len + 1);
    if (!buf) return -1;
    memcpy(buf, text, text_len + 1);

    char* words[GL_MAX_PRODUCTION_WORDS];
    uint32_t word_count = tokenize_text(buf, words, GL_MAX_PRODUCTION_WORDS);

    /* Allocate result arrays */
    result->activated_concepts = (uint64_t*)nimcp_calloc(GL_MAX_ACTIVE_CONCEPTS, sizeof(uint64_t));
    result->activation_levels = (float*)nimcp_calloc(GL_MAX_ACTIVE_CONCEPTS, sizeof(float));
    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!result->activated_concepts || !result->activation_levels || !result->semantic_vector) {
        nimcp_free(buf);
        gl_comprehension_result_cleanup(result);
        return -1;
    }

    /* Process each word */
    float total_activation = 0.0f;
    uint32_t known_words = 0;
    uint32_t concept_count = 0;

    for (uint32_t w = 0; w < word_count; w++) {
        const gl_lexicon_entry_t* entry = lexicon_find(gl, words[w]);
        if (!entry) {
            /* Unknown word - note novelty but continue */
            continue;
        }

        known_words++;

        /* Activate all bound concepts */
        for (uint32_t b = 0; b < entry->binding_count && concept_count < GL_MAX_ACTIVE_CONCEPTS; b++) {
            const gl_word_binding_t* binding = &entry->bindings[b];
            if (binding->strength < GL_ASSOC_PRUNE_THRESHOLD) continue;

            /* Check if concept already activated */
            bool found = false;
            for (uint32_t c = 0; c < concept_count; c++) {
                if (result->activated_concepts[c] == binding->concept_id) {
                    result->activation_levels[c] += binding->strength;
                    found = true;
                    break;
                }
            }

            if (!found) {
                result->activated_concepts[concept_count] = binding->concept_id;
                result->activation_levels[concept_count] = binding->strength;
                concept_count++;
            }

            total_activation += binding->strength;

            /* Add concept features to semantic vector */
            const float* cf = get_concept_features(gl, binding->concept_id);
            if (cf) {
                float weight = binding->strength / (float)word_count;
                for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                    result->semantic_vector[d] += cf[d] * weight;
                }
            }
        }

        /* Also blend in context vector (distributional) */
        if (entry->context_initialized) {
            float ctx_weight = 0.3f / (float)word_count;
            for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                result->semantic_vector[d] += entry->context_vector[d] * ctx_weight;
            }
        }
    }

    result->concept_count = concept_count;
    result->comprehension_confidence = (word_count > 0) ?
        (float)known_words / (float)word_count : 0.0f;
    result->novelty = (word_count > 0) ?
        1.0f - (float)known_words / (float)word_count : 1.0f;

    /* Normalize semantic vector */
    normalize_vector(result->semantic_vector, gl->semantic_dim);

    /* Update context */
    float blend = 0.7f; /* Favor new input */
    for (uint32_t d = 0; d < gl->semantic_dim; d++) {
        gl->context.context_vector[d] =
            blend * result->semantic_vector[d] +
            (1.0f - blend) * gl->context.context_vector[d];
    }
    gl->context.words_in_context += word_count;

    /* Store recent concepts */
    gl->context.last_concept_count = 0;
    for (uint32_t c = 0; c < concept_count && c < 16; c++) {
        gl->context.last_concepts[c] = result->activated_concepts[c];
        gl->context.last_concept_count++;
    }

    gl->stats.total_comprehensions++;

    nimcp_free(buf);
    return 0;
}

void gl_comprehension_result_cleanup(gl_comprehension_result_t* result) {
    if (!result) return;
    nimcp_free(result->activated_concepts);
    nimcp_free(result->activation_levels);
    nimcp_free(result->semantic_vector);
    memset(result, 0, sizeof(*result));
}

/*=============================================================================
 * Grounding (Cross-modal binding)
 *===========================================================================*/

int grounded_language_ground(grounded_language_t* gl, const gl_grounding_event_t* event) {
    if (!gl || !event || !event->word || !event->sensory_features) return -1;

    /* Find or create lexicon entry */
    gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, event->word);
    if (!entry) return -1;

    /* Find or create concept from sensory features */
    uint64_t concept_id = find_or_create_concept(gl,
        event->sensory_features, event->feature_dim,
        event->word, CONCEPT_OBJECT);

    if (concept_id == 0) return -1;

    /* Hebbian binding: word <-> concept, weighted by attention */
    float bind_strength = event->attention * gl->hebbian_lr;

    /* Emotional modulation: emotionally salient events create stronger bindings */
    float emotional_boost = 1.0f + fabsf(event->emotional_valence) * event->emotional_arousal;
    bind_strength *= emotional_boost;

    int rc = lexicon_bind(gl, entry, concept_id, bind_strength, event->modality);

    /* Update emotional grounding */
    float lr = gl->hebbian_lr;
    entry->valence += lr * (event->emotional_valence - entry->valence);
    entry->arousal += lr * (event->emotional_arousal - entry->arousal);
    entry->frequency++;

    /* Learn from context sentence if provided */
    if (event->context_sentence) {
        grounded_language_learn_from_text(gl, event->context_sentence);
    }

    gl->stats.total_groundings++;
    return rc;
}

int grounded_language_learn_from_text(grounded_language_t* gl, const char* text) {
    if (!gl || !text) return -1;

    size_t text_len = strlen(text);
    char* buf = (char*)nimcp_malloc(text_len + 1);
    if (!buf) return -1;
    memcpy(buf, text, text_len + 1);

    char* words[GL_MAX_PRODUCTION_WORDS];
    uint32_t word_count = tokenize_text(buf, words, GL_MAX_PRODUCTION_WORDS);

    int updates = 0;

    /* Distributional learning: co-occurring words update each other's context vectors */
    for (uint32_t i = 0; i < word_count; i++) {
        gl_lexicon_entry_t* center = lexicon_find_or_create(gl, words[i]);
        if (!center) continue;

        center->frequency++;

        /* Update context vector from surrounding words */
        uint32_t window_start = (i > GL_CONTEXT_WINDOW) ? (i - GL_CONTEXT_WINDOW) : 0;
        uint32_t window_end = (i + GL_CONTEXT_WINDOW < word_count) ?
                               (i + GL_CONTEXT_WINDOW) : word_count;

        for (uint32_t j = window_start; j < window_end; j++) {
            if (j == i) continue;

            const gl_lexicon_entry_t* neighbor = lexicon_find(gl, words[j]);
            if (!neighbor || !neighbor->context_initialized) continue;

            /* Blend neighbor's context into center's context */
            float dist_weight = 1.0f / (float)(abs((int)j - (int)i));
            float lr = gl->hebbian_lr * 0.1f * dist_weight;

            for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                center->context_vector[d] += lr * neighbor->context_vector[d];
            }
        }

        /* Initialize context if first time in context */
        if (!center->context_initialized && word_count > 2) {
            /* Random init biased by position features */
            for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                center->context_vector[d] = (gl_random(gl) - 0.5f) * 0.1f;
            }
            center->context_initialized = true;
        }

        /* Simple word class inference from position */
        if (center->learned_class == GL_CLASS_UNKNOWN && center->class_confidence < 0.5f) {
            /* Heuristic: first content word often noun, after noun often verb */
            if (i == 0 || (i > 0 && lexicon_find(gl, words[i-1]) &&
                ((const gl_lexicon_entry_t*)lexicon_find(gl, words[i-1]))->learned_class == GL_CLASS_FUNCTION)) {
                center->learned_class = GL_CLASS_NOUN;
                center->class_confidence = 0.3f;
            } else if (i > 0 && lexicon_find(gl, words[i-1]) &&
                       ((const gl_lexicon_entry_t*)lexicon_find(gl, words[i-1]))->learned_class == GL_CLASS_NOUN) {
                center->learned_class = GL_CLASS_VERB;
                center->class_confidence = 0.3f;
            }
        }

        updates++;
    }

    /* Cross-bind words in same sentence to each other's concepts (distributional grounding) */
    for (uint32_t i = 0; i < word_count; i++) {
        const gl_lexicon_entry_t* entry_i = lexicon_find(gl, words[i]);
        if (!entry_i) continue;

        for (uint32_t b = 0; b < entry_i->binding_count; b++) {
            uint64_t concept = entry_i->bindings[b].concept_id;
            float strength = entry_i->bindings[b].strength;

            /* Weakly bind co-occurring words to this concept */
            for (uint32_t j = 0; j < word_count; j++) {
                if (j == i) continue;
                gl_lexicon_entry_t* entry_j = lexicon_find_or_create(gl, words[j]);
                if (!entry_j) continue;
                if (entry_j->learned_class == GL_CLASS_FUNCTION) continue; /* Skip function words */

                lexicon_bind(gl, entry_j, concept, strength * 0.05f, GL_MODALITY_LINGUISTIC);
            }
        }
    }

    nimcp_free(buf);
    return updates;
}

uint64_t grounded_language_fast_map(grounded_language_t* gl, const char* word,
                                     const float* concept_features, uint32_t feature_dim,
                                     uint32_t category) {
    if (!gl || !word || !concept_features) return 0;

    gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, word);
    if (!entry) return 0;

    uint64_t concept_id = find_or_create_concept(gl, concept_features, feature_dim,
                                                  word, category);
    if (concept_id == 0) return 0;

    /* Strong initial binding (fast mapping) */
    lexicon_bind(gl, entry, concept_id, GL_FAST_MAP_THRESHOLD, GL_MODALITY_LINGUISTIC);

    /* Copy features to context vector for distributional similarity */
    uint32_t copy_dim = (feature_dim < gl->semantic_dim) ? feature_dim : gl->semantic_dim;
    memcpy(entry->context_vector, concept_features, copy_dim * sizeof(float));
    entry->context_initialized = true;

    LOG_DEBUG(LOG_MODULE, "Fast-mapped '%s' -> concept %lu", word, (unsigned long)concept_id);

    return concept_id;
}

/*=============================================================================
 * Production
 *===========================================================================*/

int grounded_language_produce(grounded_language_t* gl, const float* intent,
                               uint32_t intent_dim, gl_production_mode_t mode,
                               gl_production_result_t* result) {
    if (!gl || !intent || !result) return -1;
    memset(result, 0, sizeof(*result));

    /* Allocate output buffer */
    size_t buf_size = GL_MAX_PRODUCTION_WORDS * GL_MAX_WORD_LEN;
    char* buf = (char*)nimcp_calloc(buf_size, 1);
    if (!buf) return -1;

    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!result->semantic_vector) {
        nimcp_free(buf);
        return -1;
    }

    /* Find words close to the intent vector */
    const char* nouns[16] = {0};
    float noun_scores[16] = {0};
    const char* verbs[8] = {0};
    float verb_scores[8] = {0};
    const char* adjs[8] = {0};
    float adj_scores[8] = {0};

    uint32_t nn = find_words_near_vector(gl, intent, intent_dim, GL_CLASS_NOUN,
                                          nouns, noun_scores, 16);
    uint32_t nv = find_words_near_vector(gl, intent, intent_dim, GL_CLASS_VERB,
                                          verbs, verb_scores, 8);
    uint32_t na = find_words_near_vector(gl, intent, intent_dim, GL_CLASS_ADJECTIVE,
                                          adjs, adj_scores, 8);

    /* Also find any-class words as fallback */
    const char* any_words[16] = {0};
    float any_scores[16] = {0};
    uint32_t n_any = find_words_near_vector(gl, intent, intent_dim, GL_CLASS_UNKNOWN,
                                             any_words, any_scores, 16);

    /* Select template */
    const gl_template_t* tmpl = select_template(gl, nn > 0, nv > 0, na > 0);

    size_t pos = 0;
    uint32_t words_written = 0;

    if (tmpl) {
        /* Fill template slots */
        uint32_t noun_idx = 0, verb_idx = 0, adj_idx = 0;

        for (uint32_t s = 0; s < tmpl->slot_count; s++) {
            const char* word = NULL;

            switch (tmpl->slots[s]) {
                case GL_CLASS_NOUN:
                    if (noun_idx < nn) word = nouns[noun_idx++];
                    break;
                case GL_CLASS_VERB:
                    if (verb_idx < nv) word = verbs[verb_idx++];
                    break;
                case GL_CLASS_ADJECTIVE:
                    if (adj_idx < na) word = adjs[adj_idx++];
                    break;
                case GL_CLASS_FUNCTION:
                    if (tmpl->type == GL_PATTERN_COPULA) {
                        word = get_copula();
                    } else {
                        word = get_determiner(gl);
                    }
                    break;
                default:
                    break;
            }

            if (word) {
                /* Add determiner before nouns */
                if (tmpl->slots[s] == GL_CLASS_NOUN && s == 0) {
                    const char* det = get_determiner(gl);
                    size_t det_len = strlen(det);
                    if (pos + det_len + 1 < buf_size) {
                        memcpy(buf + pos, det, det_len);
                        pos += det_len;
                        buf[pos++] = ' ';
                        words_written++;
                    }
                }

                size_t wlen = strlen(word);
                if (pos + wlen + 1 < buf_size) {
                    memcpy(buf + pos, word, wlen);
                    pos += wlen;
                    buf[pos++] = ' ';
                    words_written++;
                }
            }
        }
    } else {
        /* No template — just emit top words by score */
        uint32_t max_emit = (mode == GL_PRODUCE_ELABORATE) ? 12 : 6;
        for (uint32_t i = 0; i < n_any && i < max_emit; i++) {
            if (!any_words[i]) continue;
            size_t wlen = strlen(any_words[i]);
            if (pos + wlen + 1 < buf_size) {
                memcpy(buf + pos, any_words[i], wlen);
                pos += wlen;
                buf[pos++] = ' ';
                words_written++;
            }
        }
    }

    /* Handle multi-sentence modes */
    if (mode == GL_PRODUCE_ELABORATE && tmpl && nn > 1) {
        /* Add a second sentence with remaining content */
        if (nn > 1 && na > 0) {
            pos += (size_t)snprintf(buf + pos, buf_size - pos, "The %s is %s. ",
                                     (nn > 1) ? nouns[1] : nouns[0],
                                     adjs[0]);
            words_written += 4;
        }
    }

    /* Trim trailing space */
    if (pos > 0 && buf[pos - 1] == ' ') {
        buf[pos - 1] = '.';
    }
    buf[pos] = '\0';

    /* Calculate quality metrics */
    result->text = buf;
    result->word_count = words_written;
    result->fluency = (tmpl) ? tmpl->confidence : 0.3f;
    result->relevance = (n_any > 0) ? any_scores[0] : 0.0f;
    result->creativity = 0.0f; /* Base production isn't creative */

    /* Store semantic meaning of what was produced */
    uint32_t copy_dim = (intent_dim < gl->semantic_dim) ? intent_dim : gl->semantic_dim;
    memcpy(result->semantic_vector, intent, copy_dim * sizeof(float));

    gl->stats.total_productions++;
    return 0;
}

int grounded_language_describe_concept(grounded_language_t* gl, uint64_t concept_id,
                                        gl_production_result_t* result) {
    if (!gl || !result) return -1;

    const float* features = get_concept_features(gl, concept_id);
    if (!features) {
        /* Try to describe from word bindings alone */
        const char* word = best_word_for_concept(gl, concept_id);
        if (word) {
            result->text = (char*)nimcp_malloc(strlen(word) + 16);
            if (result->text) {
                sprintf(result->text, "It is %s.", word);
                result->word_count = 3;
                result->fluency = 0.5f;
            }
            return 0;
        }
        return -1;
    }

    return grounded_language_produce(gl, features, gl->semantic_dim,
                                      GL_PRODUCE_DESCRIBE, result);
}

int grounded_language_blend(grounded_language_t* gl,
                             uint64_t concept_a, uint64_t concept_b,
                             const float* vector_a, const float* vector_b,
                             uint32_t vec_dim, float blend_ratio,
                             gl_production_result_t* result) {
    if (!gl || !result) return -1;

    uint32_t dim = gl->semantic_dim;

    /* Get feature vectors */
    const float* fa = vector_a;
    const float* fb = vector_b;

    if (concept_a != 0) fa = get_concept_features(gl, concept_a);
    if (concept_b != 0) fb = get_concept_features(gl, concept_b);

    if (!fa && !fb) return -1;

    /* Blend vectors */
    float* blended = (float*)nimcp_calloc(dim, sizeof(float));
    if (!blended) return -1;

    if (fa && fb) {
        uint32_t use_dim = (vec_dim > 0 && vec_dim < dim) ? vec_dim : dim;
        for (uint32_t i = 0; i < use_dim; i++) {
            blended[i] = (1.0f - blend_ratio) * fa[i] + blend_ratio * fb[i];
        }
    } else if (fa) {
        memcpy(blended, fa, dim * sizeof(float));
    } else {
        uint32_t copy = (vec_dim < dim) ? vec_dim : dim;
        memcpy(blended, fb, copy * sizeof(float));
    }

    int rc = grounded_language_produce(gl, blended, dim, GL_PRODUCE_CREATE, result);
    result->creativity = 0.5f + 0.5f * fabsf(blend_ratio - 0.5f); /* More creative at extremes */

    nimcp_free(blended);
    return rc;
}

int grounded_language_narrate(grounded_language_t* gl, uint64_t seed_concept,
                               uint32_t num_sentences, float creativity,
                               gl_production_result_t* result) {
    if (!gl || !result || num_sentences == 0) return -1;
    memset(result, 0, sizeof(*result));

    if (num_sentences > 10) num_sentences = 10; /* Cap for sanity */

    size_t buf_size = num_sentences * 256;
    char* buf = (char*)nimcp_calloc(buf_size, 1);
    if (!buf) return -1;

    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!result->semantic_vector) {
        nimcp_free(buf);
        return -1;
    }

    size_t pos = 0;
    uint32_t total_words = 0;
    uint64_t current_concept = seed_concept;

    for (uint32_t s = 0; s < num_sentences; s++) {
        /* Get features for current concept */
        const float* features = get_concept_features(gl, current_concept);
        if (!features) break;

        /* Produce a sentence for this concept */
        gl_production_result_t sentence = {0};
        gl_production_mode_t mode = (s == 0) ? GL_PRODUCE_DESCRIBE : GL_PRODUCE_ELABORATE;
        if (grounded_language_produce(gl, features, gl->semantic_dim, mode, &sentence) == 0) {
            if (sentence.text) {
                size_t slen = strlen(sentence.text);
                if (pos + slen + 2 < buf_size) {
                    memcpy(buf + pos, sentence.text, slen);
                    pos += slen;
                    buf[pos++] = ' ';
                    total_words += sentence.word_count;
                }
            }
            gl_production_result_cleanup(&sentence);
        }

        /* Navigate to related concept */
        if (gl->semantic_memory) {
            /* Spread activation from current concept to find next */
            semantic_memory_system_t* sm = gl->semantic_memory;
            uint64_t next_concept = 0;
            float best_score = -1.0f;

            for (uint32_t r = 0; r < sm->relation_count; r++) {
                if (!sm->relations[r]) continue;
                const semantic_relation_t* rel = sm->relations[r];

                uint64_t candidate = 0;
                if (rel->source_concept_id == current_concept) {
                    candidate = rel->target_concept_id;
                } else if (rel->target_concept_id == current_concept) {
                    candidate = rel->source_concept_id;
                }

                if (candidate != 0) {
                    float score = rel->strength;
                    /* Add randomness for creativity */
                    score += creativity * (gl_random(gl) - 0.5f) * 0.5f;
                    if (score > best_score) {
                        best_score = score;
                        next_concept = candidate;
                    }
                }
            }

            if (next_concept != 0) {
                current_concept = next_concept;
            }
        }
    }

    if (pos > 0 && buf[pos - 1] == ' ') pos--;
    buf[pos] = '\0';

    result->text = buf;
    result->word_count = total_words;
    result->fluency = (total_words > 0) ? 0.6f : 0.0f;
    result->relevance = 0.5f;
    result->creativity = creativity;

    return 0;
}

void gl_production_result_cleanup(gl_production_result_t* result) {
    if (!result) return;
    nimcp_free(result->text);
    nimcp_free(result->semantic_vector);
    memset(result, 0, sizeof(*result));
}

/*=============================================================================
 * Conversation
 *===========================================================================*/

int grounded_language_respond(grounded_language_t* gl, const char* input_text,
                               char* response, uint32_t response_max, float* confidence) {
    if (!gl || !input_text || !response || response_max == 0) return -1;

    /* Step 1: Comprehend input */
    gl_comprehension_result_t comp = {0};
    if (grounded_language_comprehend(gl, input_text, &comp) != 0) {
        gl_comprehension_result_cleanup(&comp);
        return -1;
    }

    /* Step 2: Produce response from comprehended meaning */
    gl_production_result_t prod = {0};
    int rc = -1;

    if (comp.semantic_vector && comp.comprehension_confidence > 0.1f) {
        rc = grounded_language_produce(gl, comp.semantic_vector, gl->semantic_dim,
                                        GL_PRODUCE_RESPOND, &prod);
    }

    if (rc != 0 || !prod.text || prod.word_count == 0) {
        /* Fallback: echo what we understood */
        if (comp.concept_count > 0) {
            const char* word = best_word_for_concept(gl, comp.activated_concepts[0]);
            if (word) {
                snprintf(response, response_max, "I understand about %s.", word);
            } else {
                snprintf(response, response_max, "I am learning.");
            }
        } else {
            snprintf(response, response_max, "I do not understand yet.");
        }
        if (confidence) *confidence = 0.1f;
    } else {
        strncpy(response, prod.text, response_max - 1);
        response[response_max - 1] = '\0';
        if (confidence) *confidence = prod.fluency * prod.relevance;
    }

    int result_len = (int)strlen(response);

    gl_comprehension_result_cleanup(&comp);
    gl_production_result_cleanup(&prod);

    return result_len;
}

/*=============================================================================
 * Training / Learning
 *===========================================================================*/

float grounded_language_learn_pair(grounded_language_t* gl, const char* input_text,
                                    const char* target_text, float learning_rate) {
    if (!gl || !input_text || !target_text) return -1.0f;

    float lr = (learning_rate > 0.0f) ? learning_rate : gl->hebbian_lr;
    float saved_lr = gl->hebbian_lr;
    gl->hebbian_lr = lr;

    /* Comprehend input */
    gl_comprehension_result_t input_comp = {0};
    grounded_language_comprehend(gl, input_text, &input_comp);

    /* Comprehend target */
    gl_comprehension_result_t target_comp = {0};
    grounded_language_comprehend(gl, target_text, &target_comp);

    /* Learn distributional patterns from both texts */
    grounded_language_learn_from_text(gl, input_text);
    grounded_language_learn_from_text(gl, target_text);

    /* Cross-bind: input concepts should be associated with target words and vice versa */
    /* This teaches the brain that hearing X should lead to saying Y */

    /* Tokenize target for word-level binding */
    size_t tlen = strlen(target_text);
    char* tbuf = (char*)nimcp_malloc(tlen + 1);
    if (tbuf) {
        memcpy(tbuf, target_text, tlen + 1);
        char* twords[GL_MAX_PRODUCTION_WORDS];
        uint32_t tword_count = tokenize_text(tbuf, twords, GL_MAX_PRODUCTION_WORDS);

        /* For each input concept, bind to target words */
        for (uint32_t c = 0; c < input_comp.concept_count; c++) {
            uint64_t concept = input_comp.activated_concepts[c];
            float activation = input_comp.activation_levels[c];

            for (uint32_t w = 0; w < tword_count; w++) {
                gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, twords[w]);
                if (!entry) continue;
                if (entry->learned_class == GL_CLASS_FUNCTION) continue;

                lexicon_bind(gl, entry, concept, activation * lr * 0.5f, GL_MODALITY_LINGUISTIC);
            }
        }

        /* For each target concept, bind to input words */
        size_t ilen = strlen(input_text);
        char* ibuf = (char*)nimcp_malloc(ilen + 1);
        if (ibuf) {
            memcpy(ibuf, input_text, ilen + 1);
            char* iwords[GL_MAX_PRODUCTION_WORDS];
            uint32_t iword_count = tokenize_text(ibuf, iwords, GL_MAX_PRODUCTION_WORDS);

            for (uint32_t c = 0; c < target_comp.concept_count; c++) {
                uint64_t concept = target_comp.activated_concepts[c];
                float activation = target_comp.activation_levels[c];

                for (uint32_t w = 0; w < iword_count; w++) {
                    gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, iwords[w]);
                    if (!entry) continue;
                    if (entry->learned_class == GL_CLASS_FUNCTION) continue;

                    lexicon_bind(gl, entry, concept, activation * lr * 0.5f, GL_MODALITY_LINGUISTIC);
                }
            }
            nimcp_free(ibuf);
        }

        nimcp_free(tbuf);
    }

    /* Learn syntactic patterns from target */
    grounded_language_learn_syntax(gl, target_text);

    /* Compute loss: cosine distance between input semantics and target semantics */
    float loss = 1.0f;
    if (input_comp.semantic_vector && target_comp.semantic_vector) {
        float sim = cosine_similarity(input_comp.semantic_vector, target_comp.semantic_vector,
                                       gl->semantic_dim);
        loss = 1.0f - sim; /* 0 = perfect alignment, 1 = orthogonal */
    }

    gl->hebbian_lr = saved_lr;
    gl_comprehension_result_cleanup(&input_comp);
    gl_comprehension_result_cleanup(&target_comp);

    return loss;
}

int grounded_language_learn_syntax(grounded_language_t* gl, const char* text) {
    if (!gl || !text) return -1;

    size_t tlen = strlen(text);
    char* buf = (char*)nimcp_malloc(tlen + 1);
    if (!buf) return -1;
    memcpy(buf, text, tlen + 1);

    char* words[GL_MAX_PRODUCTION_WORDS];
    uint32_t word_count = tokenize_text(buf, words, GL_MAX_PRODUCTION_WORDS);

    if (word_count < 2) {
        nimcp_free(buf);
        return 0;
    }

    /* Extract word class sequence */
    gl_word_class_t classes[GL_MAX_TEMPLATE_SLOTS];
    uint32_t class_count = 0;

    for (uint32_t i = 0; i < word_count && class_count < GL_MAX_TEMPLATE_SLOTS; i++) {
        const gl_lexicon_entry_t* entry = lexicon_find(gl, words[i]);
        if (!entry) {
            classes[class_count++] = GL_CLASS_UNKNOWN;
            continue;
        }

        /* Skip function words in the class sequence (they're structural) */
        if (entry->learned_class == GL_CLASS_FUNCTION ||
            entry->learned_class == GL_CLASS_PRONOUN) {
            continue;
        }

        classes[class_count++] = entry->learned_class;
    }

    /* Try to match against existing templates */
    int new_patterns = 0;
    bool matched = false;

    for (uint32_t t = 0; t < gl->template_count; t++) {
        gl_template_t* tmpl = &gl->templates[t];
        if (tmpl->slot_count != class_count) continue;

        bool match = true;
        for (uint32_t s = 0; s < class_count; s++) {
            if (tmpl->slots[s] != classes[s] && classes[s] != GL_CLASS_UNKNOWN) {
                match = false;
                break;
            }
        }

        if (match) {
            tmpl->frequency += 1.0f;
            tmpl->confidence = 1.0f - expf(-tmpl->frequency / 10.0f);
            matched = true;
            break;
        }
    }

    /* Create new template if no match and we have a clear pattern */
    if (!matched && class_count >= 2 && class_count <= GL_MAX_TEMPLATE_SLOTS &&
        gl->template_count < gl->template_capacity) {

        bool has_known = false;
        for (uint32_t s = 0; s < class_count; s++) {
            if (classes[s] != GL_CLASS_UNKNOWN) has_known = true;
        }

        if (has_known) {
            gl_template_t* tmpl = &gl->templates[gl->template_count++];
            tmpl->type = GL_PATTERN_LEARNED;
            for (uint32_t s = 0; s < class_count; s++) {
                tmpl->slots[s] = classes[s];
            }
            tmpl->slot_count = class_count;
            tmpl->frequency = 1.0f;
            tmpl->confidence = 0.2f;
            gl->stats.templates_learned++;
            new_patterns++;
        }
    }

    nimcp_free(buf);
    return new_patterns;
}

/*=============================================================================
 * Cross-modal Connections
 *===========================================================================*/

void grounded_language_connect_visual(grounded_language_t* gl, void* vis_ctx) {
    if (gl) gl->visual_ctx = vis_ctx;
}

void grounded_language_connect_auditory(grounded_language_t* gl, void* aud_ctx) {
    if (gl) gl->auditory_ctx = aud_ctx;
}

void grounded_language_connect_speech(grounded_language_t* gl, void* speech_ctx) {
    if (gl) gl->speech_ctx = speech_ctx;
}

void grounded_language_connect_columns(grounded_language_t* gl, void* col_pool) {
    if (gl) gl->column_pool = col_pool;
}

void grounded_language_connect_emotional(grounded_language_t* gl, void* emo_ctx) {
    if (gl) gl->emotional_ctx = emo_ctx;
}

/*=============================================================================
 * Query / Introspection
 *===========================================================================*/

const gl_lexicon_entry_t* grounded_language_lookup(const grounded_language_t* gl,
                                                     const char* word) {
    if (!gl || !word) return NULL;
    return lexicon_find(gl, word);
}

uint32_t grounded_language_words_for_concept(const grounded_language_t* gl,
                                              uint64_t concept_id,
                                              const char** words,
                                              uint32_t max_words) {
    if (!gl || !words || max_words == 0) return 0;

    uint32_t count = 0;
    for (uint32_t w = 0; w < gl->vocab_count && count < max_words; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry) continue;

        for (uint32_t b = 0; b < entry->binding_count; b++) {
            if (entry->bindings[b].concept_id == concept_id &&
                entry->bindings[b].strength > GL_ASSOC_PRUNE_THRESHOLD) {
                words[count++] = entry->form;
                break;
            }
        }
    }

    return count;
}

void grounded_language_get_stats(const grounded_language_t* gl, gl_stats_t* stats) {
    if (!gl || !stats) return;
    *stats = gl->stats;

    /* Compute average binding strength */
    float total_strength = 0.0f;
    uint32_t total_bindings = 0;
    for (uint32_t w = 0; w < gl->vocab_count; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry) continue;
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            total_strength += entry->bindings[b].strength;
            total_bindings++;
        }
    }
    stats->avg_binding_strength = (total_bindings > 0) ?
        total_strength / (float)total_bindings : 0.0f;
}

/*=============================================================================
 * Serialization (stub — full implementation deferred)
 *===========================================================================*/

int grounded_language_save(const grounded_language_t* gl, const char* path) {
    if (!gl || !path) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    /* Magic + version */
    uint32_t magic = GL_MAGIC;
    uint32_t version = 1;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&gl->semantic_dim, sizeof(gl->semantic_dim), 1, f);
    fwrite(&gl->vocab_count, sizeof(gl->vocab_count), 1, f);

    /* Write each word entry */
    for (uint32_t w = 0; w < gl->vocab_count; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry) continue;

        /* Word form */
        fwrite(entry->form, GL_MAX_WORD_LEN, 1, f);
        fwrite(&entry->frequency, sizeof(entry->frequency), 1, f);
        fwrite(&entry->learned_class, sizeof(entry->learned_class), 1, f);
        fwrite(&entry->valence, sizeof(entry->valence), 1, f);
        fwrite(&entry->arousal, sizeof(entry->arousal), 1, f);

        /* Context vector */
        fwrite(&entry->context_initialized, sizeof(entry->context_initialized), 1, f);
        if (entry->context_initialized) {
            fwrite(entry->context_vector, sizeof(float), gl->semantic_dim, f);
        }

        /* Bindings */
        fwrite(&entry->binding_count, sizeof(entry->binding_count), 1, f);
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            fwrite(&entry->bindings[b], sizeof(gl_word_binding_t), 1, f);
        }
    }

    /* Templates */
    fwrite(&gl->template_count, sizeof(gl->template_count), 1, f);
    fwrite(gl->templates, sizeof(gl_template_t), gl->template_count, f);

    fclose(f);
    LOG_INFO(LOG_MODULE, "Saved grounded language state to %s (%u words, %u templates)",
             path, gl->vocab_count, gl->template_count);
    return 0;
}

grounded_language_t* grounded_language_load(const char* path, void* semantic_memory) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic, version, semantic_dim, vocab_count;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != GL_MAGIC) {
        fclose(f);
        return NULL;
    }
    fread(&version, sizeof(version), 1, f);
    fread(&semantic_dim, sizeof(semantic_dim), 1, f);
    fread(&vocab_count, sizeof(vocab_count), 1, f);

    grounded_language_t* gl = grounded_language_create(semantic_dim, semantic_memory);
    if (!gl) { fclose(f); return NULL; }

    /* Read word entries */
    for (uint32_t w = 0; w < vocab_count; w++) {
        char form[GL_MAX_WORD_LEN];
        uint32_t frequency;
        gl_word_class_t word_class;
        float valence, arousal;
        bool ctx_init;

        fread(form, GL_MAX_WORD_LEN, 1, f);
        fread(&frequency, sizeof(frequency), 1, f);
        fread(&word_class, sizeof(word_class), 1, f);
        fread(&valence, sizeof(valence), 1, f);
        fread(&arousal, sizeof(arousal), 1, f);

        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, form);
        if (!entry) continue;

        entry->frequency = frequency;
        entry->learned_class = word_class;
        entry->valence = valence;
        entry->arousal = arousal;

        fread(&ctx_init, sizeof(ctx_init), 1, f);
        entry->context_initialized = ctx_init;
        if (ctx_init) {
            fread(entry->context_vector, sizeof(float), semantic_dim, f);
        }

        uint32_t binding_count;
        fread(&binding_count, sizeof(binding_count), 1, f);

        for (uint32_t b = 0; b < binding_count; b++) {
            gl_word_binding_t binding;
            fread(&binding, sizeof(binding), 1, f);

            /* Grow bindings array if needed */
            if (entry->binding_count >= entry->binding_capacity) {
                uint32_t new_cap = entry->binding_capacity * 2;
                gl_word_binding_t* new_arr = (gl_word_binding_t*)nimcp_realloc(
                    entry->bindings, new_cap * sizeof(gl_word_binding_t));
                if (!new_arr) continue;
                entry->bindings = new_arr;
                entry->binding_capacity = new_cap;
            }
            entry->bindings[entry->binding_count++] = binding;
        }
    }

    /* Templates */
    uint32_t template_count;
    fread(&template_count, sizeof(template_count), 1, f);
    if (template_count <= gl->template_capacity) {
        fread(gl->templates, sizeof(gl_template_t), template_count, f);
        gl->template_count = template_count;
    }

    fclose(f);
    LOG_INFO(LOG_MODULE, "Loaded grounded language state from %s (%u words)", path, gl->vocab_count);
    return gl;
}
