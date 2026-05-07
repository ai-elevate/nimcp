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
#include "language/nimcp_grounded_language_internal.h"
#include "language/nimcp_bigram_spectrum.h"
/* PA-4+ : compile the bigram-spectrum implementation as part of this
 * translation unit. The build's source list is explicit (not GLOB) and
 * the CMakeLists.txt is outside the modify set for this change, so we
 * inline-include the .c here — same pattern as nimcp.c → nimcp_part_*.c.
 * The header (nimcp_bigram_spectrum.h) provides the public API and is
 * included separately by the brain wrapper and the test. */
#include "nimcp_bigram_spectrum.c"
#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "cognitive/memory/nimcp_semantic_memory.h"
#include "cognitive/memory/nimcp_engram.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "security/nimcp_bbb_helpers.h"

#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>  /* g_spectrum_map_lock */

/* Silence -Wunused-result for fread — reads are best-effort; file-format magic
 * and version checks downstream validate integrity. */
static inline void nimcp_fread_ignore(void* ptr, size_t sz, size_t n, FILE* f) {
    size_t got = fread(ptr, sz, n, f);
    (void)got;
}
#define fread_chk nimcp_fread_ignore

#define LOG_MODULE "GROUNDED_LANG"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(grounded_language)

/*=============================================================================
 * Internal Structures
 *
 * `struct grounded_language` and `gl_context_t` are defined in
 * src/language/nimcp_grounded_language_internal.h so that the persistence
 * sidecar (nimcp_grounded_language_persistence.c) can iterate the lexicon
 * field-by-field without duplicating the layout. Do NOT add struct
 * definitions here — put them in the internal header.
 *===========================================================================*/

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

            /* NLP: seed learned_class from morphological cue if any.
             * Low confidence (0.4) — surface morphology is suggestive
             * but not authoritative; positional context will overwrite. */
            extern gl_word_class_t gl_morph_pos_hint(const char*);
            gl_word_class_t hint = gl_morph_pos_hint(entry->form);
            if (hint != GL_CLASS_UNKNOWN) {
                entry->learned_class = hint;
                entry->class_confidence = 0.4f;
            }

            /* Mirror the new entry into broca/wernicke region lexicons
             * if either adapter is connected. No-op when not wired. */
            extern void gl_mirror_new_word_to_regions(grounded_language_t*,
                                                       const char*);
            gl_mirror_new_word_to_regions(gl, entry->form);

            /* Fire NEW_WORD event on the cognitive bus.
             *
             * D7: suppress during persistence rehydrate. Resume blasts
             * ~30K NEW_WORD events to every subscriber (inner-speech,
             * imagination, theory-of-mind, empathy, introspection,
             * prefrontal, insula, cingulate, amygdala, ofc, broca,
             * wernicke, hippocampus, ...) — pure noise on cold boot.
             * Live grounding events still fire normally. */
            if (!gl->is_loading) {
                extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
                gl_event_t ev = {0};
                ev.type = GL_EVENT_NEW_WORD;
                ev.word = entry->form;
                gl_fire_event(gl, &ev);
            }

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

/* External alias used by the NLP frontend (sibling .c file). */
const gl_lexicon_entry_t* lexicon_find_internal(const grounded_language_t* gl,
                                                  const char* word) {
    return lexicon_find(gl, word);
}

/* Public read-only "is this word in the lexicon?" probe (#1).
 * Used by region adapters (broca/wernicke) to decide whether to fall
 * through to GL when their local lexicons miss, without paying the
 * cost of a full comprehend / fuzzy / morph chain. */
bool grounded_language_has_word(const grounded_language_t* gl, const char* word) {
    if (!gl || !word || !word[0]) return false;
    return lexicon_find(gl, word) != NULL;
}

/*=============================================================================
 * #9 Compositional phrase tracking
 *===========================================================================*/

/* Build a space-joined lowercased phrase form from N word tokens.
 * Returns 1 on success, 0 on overflow. */
static int _gl_phrase_form(char* out, size_t out_sz,
                             char** words, uint32_t n) {
    if (!out || out_sz == 0 || !words || n == 0) return 0;
    size_t pos = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) {
            if (pos + 1 >= out_sz) return 0;
            out[pos++] = ' ';
        }
        for (const char* p = words[i]; *p; p++) {
            if (pos + 1 >= out_sz) return 0;
            unsigned char c = (unsigned char)*p;
            if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
            out[pos++] = (char)c;
        }
    }
    out[pos] = '\0';
    return 1;
}

/* Find phrase by form, or insert if not present. When at capacity,
 * evicts the least-frequent existing phrase (LRU-by-frequency) so
 * long-running training keeps the most-useful phrases rather than
 * locking in the first 512 ever seen. Returns the slot — never NULL
 * once gl->phrases is allocated. Component_words is recorded on
 * insert (or refresh after eviction). */
static gl_phrase_t* _gl_phrase_find_or_create(grounded_language_t* gl,
                                                const char* form,
                                                uint8_t component_words) {
    if (!gl || !gl->phrases || !form) return NULL;
    uint32_t h = hash_word(form);
    /* Linear scan — N ≤ GL_MAX_PHRASES = 512. learn_from_text is
     * cold-ish and we can afford O(N) per phrase track. */
    for (uint32_t i = 0; i < gl->phrase_count; i++) {
        if (gl->phrases[i].form_hash == h &&
            strcmp(gl->phrases[i].form, form) == 0) {
            return &gl->phrases[i];
        }
    }

    gl_phrase_t* p;
    if (gl->phrase_count < GL_MAX_PHRASES) {
        p = &gl->phrases[gl->phrase_count++];
    } else {
        /* Capacity overflow — evict least-frequent. Linear scan to
         * pick the victim. */
        uint32_t victim = 0;
        for (uint32_t i = 1; i < gl->phrase_count; i++) {
            if (gl->phrases[i].frequency < gl->phrases[victim].frequency) {
                victim = i;
            }
        }
        /* Refuse to evict if the victim has higher frequency than the
         * incoming phrase would naturally start at — the new phrase
         * starts at frequency=0 and would be evicted on its own first
         * tick. Drop the new one instead. */
        if (gl->phrases[victim].frequency > 0) {
            p = &gl->phrases[victim];
            nimcp_free(p->semantic_vec);
            memset(p, 0, sizeof(*p));
            gl->phrases_evicted++;
        } else {
            /* Everything is freq=0 — can't reliably pick a victim.
             * Reject the new phrase rather than thrash. */
            return NULL;
        }
    }

    size_t flen = strlen(form);
    if (flen >= GL_MAX_PHRASE_LEN) flen = GL_MAX_PHRASE_LEN - 1;
    memcpy(p->form, form, flen);
    p->form[flen] = '\0';
    p->form_hash = h;
    p->component_words = component_words;
    p->frequency = 0;
    p->semantic_vec = NULL;
    p->vec_initialized = false;
    return p;
}

/* Track bigrams + trigrams in a sentence. Cheap; called once per
 * learn_from_text after the existing distributional loops. */
static void _gl_track_phrases(grounded_language_t* gl,
                                char** words, uint32_t word_count) {
    if (!gl || !gl->phrases || word_count < 2) return;
    char buf[GL_MAX_PHRASE_LEN];

    for (uint32_t i = 0; i + 1 < word_count; i++) {
        if (_gl_phrase_form(buf, sizeof(buf), &words[i], 2)) {
            gl_phrase_t* p = _gl_phrase_find_or_create(gl, buf, 2);
            if (p) {
                p->frequency++;
                /* Vector cache invalidated when components change —
                 * cheapest way is to clear on any frequency tick so
                 * the next lookup recomputes from current word
                 * vectors. We could be smarter but recompute is O(dim). */
                p->vec_initialized = false;
            }
        }
        if (i + 2 < word_count) {
            if (_gl_phrase_form(buf, sizeof(buf), &words[i], 3)) {
                gl_phrase_t* p = _gl_phrase_find_or_create(gl, buf, 3);
                if (p) { p->frequency++; p->vec_initialized = false; }
            }
        }
    }
}

const gl_phrase_t* grounded_language_lookup_phrase(
        const grounded_language_t* gl, const char* form) {
    if (!gl || !gl->phrases || !form) return NULL;
    /* Lowercase the input into a stack buffer for case-insensitive lookup. */
    char low[GL_MAX_PHRASE_LEN];
    size_t i = 0;
    for (; form[i] && i < sizeof(low) - 1; i++) {
        unsigned char c = (unsigned char)form[i];
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        low[i] = (char)c;
    }
    low[i] = '\0';
    uint32_t h = hash_word(low);
    for (uint32_t k = 0; k < gl->phrase_count; k++) {
        if (gl->phrases[k].form_hash == h &&
            strcmp(gl->phrases[k].form, low) == 0) {
            /* Lazy compute the semantic vector from constituent words. */
            gl_phrase_t* mut = (gl_phrase_t*)&gl->phrases[k];
            if (!mut->vec_initialized) {
                if (!mut->semantic_vec) {
                    mut->semantic_vec = (float*)nimcp_calloc(
                        gl->semantic_dim, sizeof(float));
                    if (!mut->semantic_vec) return NULL;
                }
                memset(mut->semantic_vec, 0,
                       gl->semantic_dim * sizeof(float));
                /* Tokenize the form back out and average each
                 * component's context_vector. */
                char tmp[GL_MAX_PHRASE_LEN];
                memcpy(tmp, mut->form, sizeof(tmp));
                char* save = NULL;
                char* tok = strtok_r(tmp, " ", &save);
                uint32_t n = 0;
                while (tok) {
                    const gl_lexicon_entry_t* e = lexicon_find(gl, tok);
                    if (e && e->context_initialized) {
                        for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                            mut->semantic_vec[d] += e->context_vector[d];
                        }
                        n++;
                    }
                    tok = strtok_r(NULL, " ", &save);
                }
                if (n > 0) {
                    float inv = 1.0f / (float)n;
                    for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                        mut->semantic_vec[d] *= inv;
                    }
                }
                mut->vec_initialized = true;
            }
            return &gl->phrases[k];
        }
    }
    return NULL;
}

uint32_t grounded_language_phrase_count(const grounded_language_t* gl) {
    return gl ? gl->phrase_count : 0u;
}

uint32_t grounded_language_get_top_phrases(const grounded_language_t* gl,
                                             uint32_t min_freq,
                                             uint8_t min_n,
                                             const gl_phrase_t** out_phrases,
                                             uint32_t max_k) {
    if (!gl || !gl->phrases || !out_phrases || max_k == 0) return 0;
    uint32_t n = gl->phrase_count;
    if (n == 0) return 0;

    /* Selection sort over the first max_k positions. N ≤ 512 — cheap. */
    /* Build an index array filtered by qualifiers first. */
    uint32_t idx_buf_stack[GL_MAX_PHRASES];
    uint32_t* idx = idx_buf_stack;
    uint32_t qcount = 0;
    for (uint32_t i = 0; i < n; i++) {
        const gl_phrase_t* p = &gl->phrases[i];
        if (p->frequency < min_freq) continue;
        if (min_n != 0 && p->component_words != min_n) continue;
        idx[qcount++] = i;
    }
    if (qcount == 0) return 0;

    uint32_t k = (qcount < max_k) ? qcount : max_k;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < qcount; j++) {
            if (gl->phrases[idx[j]].frequency >
                gl->phrases[idx[best]].frequency) best = j;
        }
        if (best != i) { uint32_t t = idx[i]; idx[i] = idx[best]; idx[best] = t; }
        out_phrases[i] = &gl->phrases[idx[i]];
    }
    return k;
}

/* Per-modality binding-coverage probe. See header for contract.
 *
 * For every binding in every lexicon entry, increment out_counts[m] for
 * each modality m whose modality_strength[m] > 0. A binding grounded in
 * just visual contributes only to out_counts[VISUAL]; one reinforced
 * across visual+auditory contributes to both. This is the natural
 * "coverage" probe — it answers "how many bindings carry information in
 * modality m" rather than collapsing to a single dominant modality. */
void grounded_language_get_modality_counts(const grounded_language_t* gl,
                                            uint32_t out_counts[GL_MODALITY_COUNT]) {
    if (!out_counts) return;
    for (uint32_t m = 0; m < GL_MODALITY_COUNT; m++) out_counts[m] = 0;
    if (!gl || !gl->vocab_list || gl->vocab_count == 0) return;

    for (uint32_t v = 0; v < gl->vocab_count; v++) {
        const gl_lexicon_entry_t* e = gl->vocab_list[v];
        if (!e || !e->bindings) continue;
        for (uint32_t bi = 0; bi < e->binding_count; bi++) {
            const gl_word_binding_t* b = &e->bindings[bi];
            for (uint32_t m = 0; m < GL_MODALITY_COUNT; m++) {
                /* Defensive: GL_MODALITY_COUNT is the array length so the
                 * index is in-range by construction; the loop bound is the
                 * "modality < GL_MODALITY_COUNT" guard. Count any modality
                 * that has accrued binding strength > 0 (set on the first
                 * grounding event for that modality, decayed but never
                 * pushed below 0 by subsequent updates). */
                if (b->modality_strength[m] > 0.0f) out_counts[m]++;
            }
        }
    }
}

/*=============================================================================
 * #12 SNN spike → lexicon decoding
 *===========================================================================*/

int gl_observe_snn_spikes(grounded_language_t* gl,
                            const float* spike_rates,
                            uint32_t rate_dim,
                            float* confidence_out) {
    if (!gl || !spike_rates || rate_dim == 0) return -1;
    if (rate_dim != gl->semantic_dim) return -1;
    if (confidence_out) *confidence_out = 0.0f;
    if (gl->vocab_count == 0) return 0;

    /* Find best-matching lexicon entry by cosine similarity of
     * spike_rates against each entry's context_vector. */
    int32_t best_idx = -1;
    float best_sim = 0.0f;
    for (uint32_t v = 0; v < gl->vocab_count; v++) {
        const gl_lexicon_entry_t* e = gl->vocab_list[v];
        if (!e || !e->context_initialized) continue;
        float sim = cosine_similarity(spike_rates, e->context_vector,
                                        gl->semantic_dim);
        if (sim > best_sim) { best_sim = sim; best_idx = (int32_t)v; }
    }

    if (confidence_out) *confidence_out = best_sim;
    if (best_idx < 0 || best_sim < GL_SNN_SPIKE_MATCH_THRESHOLD) return 0;

    /* Fire COMPREHENDED for the matched word. */
    extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
    gl_event_t bus_ev = {0};
    bus_ev.type         = GL_EVENT_COMPREHENDED;
    bus_ev.word         = gl->vocab_list[best_idx]->form;
    bus_ev.text         = gl->vocab_list[best_idx]->form;
    bus_ev.semantic_vec = spike_rates;
    bus_ev.confidence   = best_sim;
    gl_fire_event(gl, &bus_ev);
    return 1;
}

/*=============================================================================
 * #8 Cross-modal disambiguation
 *===========================================================================*/

uint32_t grounded_language_disambiguate(const grounded_language_t* gl,
                                          const char* word,
                                          const float* modality_weights,
                                          uint64_t* out_concepts,
                                          float* out_scores,
                                          uint32_t max_k) {
    if (!gl || !word || !out_concepts || !out_scores || max_k == 0) return 0;

    const gl_lexicon_entry_t* e = lexicon_find(gl, word);
    if (!e || e->binding_count == 0) return 0;

    /* Determine if we have a real weight signal. NULL or all-zero
     * weights → fall back to plain confidence × overall-strength
     * ranking (general "best concept" query). */
    bool have_weights = false;
    if (modality_weights) {
        for (uint32_t m = 0; m < GL_MODALITY_COUNT; m++) {
            if (modality_weights[m] > 1e-6f) { have_weights = true; break; }
        }
    }

    /* Score each binding. */
    uint32_t n = e->binding_count;
    /* Stack-bounded — entry binding_count grows on demand but is
     * realistically <64 in practice. Use a small buffer with overflow
     * fallback to heap. */
    float  stack_scores[64];
    uint64_t stack_ids[64];
    float* scores = (n <= 64) ? stack_scores : (float*)nimcp_malloc(n * sizeof(float));
    uint64_t* ids = (n <= 64) ? stack_ids    : (uint64_t*)nimcp_malloc(n * sizeof(uint64_t));
    if (!scores || !ids) {
        if (n > 64) { nimcp_free(scores); nimcp_free(ids); }
        return 0;
    }

    for (uint32_t i = 0; i < n; i++) {
        const gl_word_binding_t* b = &e->bindings[i];
        float modality_score = 0.0f;
        if (have_weights) {
            for (uint32_t m = 0; m < GL_MODALITY_COUNT; m++) {
                float w = modality_weights[m];
                if (w < 0.0f) w = 0.0f;
                if (w > 1.0f) w = 1.0f;
                modality_score += b->modality_strength[m] * w;
            }
        } else {
            modality_score = b->strength;
        }
        /* Use confidence directly. Clamp tiny floor so brand-new
         * bindings (confidence ≈ 0.2 from lexicon_bind) still register
         * a non-trivial score, but never let a 0-confidence binding
         * outrank a real one with a 1.0 fallback. */
        float conf = b->confidence;
        if (conf < 0.05f) conf = 0.05f;
        scores[i] = modality_score * conf;
        ids[i]    = b->concept_id;
    }

    /* Partial top-k selection (selection sort over the first max_k
     * positions). N ≤ ~64 in practice — no need for heapsort. */
    uint32_t k = (n < max_k) ? n : max_k;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n; j++) {
            if (scores[j] > scores[best]) best = j;
        }
        if (best != i) {
            float ts = scores[i]; scores[i] = scores[best]; scores[best] = ts;
            uint64_t ti = ids[i]; ids[i] = ids[best]; ids[best] = ti;
        }
        out_concepts[i] = ids[i];
        out_scores[i]   = scores[i];
    }

    if (n > 64) { nimcp_free(scores); nimcp_free(ids); }
    return k;
}

/*=============================================================================
 * #5 LRU lexicon eviction
 *===========================================================================*/

/* Free a single lexicon entry's heap storage. */
static void _gl_free_entry(gl_lexicon_entry_t* e) {
    if (!e) return;
    nimcp_free(e->bindings);
    nimcp_free(e->context_vector);
    nimcp_free(e);
}

/* Rebuild the hash table from vocab_list. After eviction the previous
 * probe chains are stale; rather than chase tombstones through every
 * lookup we wipe and re-insert. O(N) — only called from the eviction
 * path which is itself rare. */
static void _gl_rebuild_lexicon_hash(grounded_language_t* gl) {
    memset(gl->lexicon, 0, gl->lexicon_size * sizeof(gl_lexicon_entry_t*));
    for (uint32_t v = 0; v < gl->vocab_count; v++) {
        gl_lexicon_entry_t* e = gl->vocab_list[v];
        if (!e) continue;
        uint32_t idx = e->form_hash % gl->lexicon_size;
        for (uint32_t probe = 0; probe < gl->lexicon_size; probe++) {
            uint32_t slot = (idx + probe) % gl->lexicon_size;
            if (!gl->lexicon[slot]) {
                gl->lexicon[slot] = e;
                break;
            }
        }
    }
}

uint32_t grounded_language_evict_lru(grounded_language_t* gl, uint32_t n) {
    if (!gl || n == 0 || gl->vocab_count == 0) return 0;

    /* Pass 1: find the n-th lowest frequency among unpinned entries.
     * For small batches an N×K linear scan is fine — N ≤ vocab_count,
     * K = n. With n ≤ ~256 and vocab ≤ 16384 we're talking 4M ops
     * worst-case, which is microseconds. No need for partial-sort. */
    uint32_t evicted = 0;
    while (evicted < n) {
        uint32_t target = UINT32_MAX;
        uint32_t target_freq = UINT32_MAX;
        for (uint32_t v = 0; v < gl->vocab_count; v++) {
            const gl_lexicon_entry_t* e = gl->vocab_list[v];
            if (!e) continue;
            if (e->frequency >= GL_LRU_FREQ_PIN_FLOOR) continue;
            if (e->frequency < target_freq) {
                target_freq = e->frequency;
                target = v;
            }
        }
        if (target == UINT32_MAX) break;  /* nothing left to evict */

        _gl_free_entry(gl->vocab_list[target]);
        gl->vocab_list[target] = NULL;
        evicted++;
    }

    if (evicted == 0) return 0;

    /* Compact vocab_list — shift survivors down to fill the holes. */
    uint32_t write = 0;
    for (uint32_t read = 0; read < gl->vocab_count; read++) {
        if (gl->vocab_list[read]) {
            if (write != read) gl->vocab_list[write] = gl->vocab_list[read];
            write++;
        }
    }
    /* Clear stale tail slots so destroy() doesn't double-free. */
    for (uint32_t i = write; i < gl->vocab_count; i++) gl->vocab_list[i] = NULL;
    gl->vocab_count = write;
    gl->stats.vocab_size = gl->vocab_count;

    /* Rebuild hash table (probe chains are now stale). */
    _gl_rebuild_lexicon_hash(gl);

    LOG_DEBUG(LOG_MODULE,
              "lexicon eviction: removed %u entries (vocab now %u)",
              evicted, gl->vocab_count);
    return evicted;
}

/*=============================================================================
 * Fuzzy Matching (Two-stage fallback for typos/misspellings)
 *===========================================================================*/

/** Minimum similarity threshold to accept a fuzzy match */
#define GL_FUZZY_THRESHOLD  0.65f

/** Sanity floor for the *weaker* of the two signals (phonological vs
 *  character-set). Typos clear both signals; random consonant strings
 *  clear phonological alone (5-feature space is small) and fail this. */
#define GL_FUZZY_MIN_SIGNAL 0.40f

/** Minimum word length to attempt fuzzy matching (skip "a", "I", etc.) */
#define GL_FUZZY_MIN_LEN    3

/**
 * Phonological similarity — distinctive feature comparison.
 *
 * Maps each letter to a simplified articulatory feature vector (place, manner,
 * voicing) and compares via feature overlap. Catches sound-alike typos:
 *   "waht" ↔ "what"  (same phonemes, transposed)
 *   "mathmatics" ↔ "mathematics"  (vowel deletion)
 *   "kalkulus" ↔ "calculus"  (phonetic spelling)
 *
 * Returns similarity in [0, 1].
 */
static float phonological_similarity(const char* a, const char* b) {
    /* Simplified phoneme feature vectors for English letters:
     * [place(0-7), manner(0-7), voicing(0/1), vowel_height(0-3), is_vowel]
     * These approximate distinctive features from phonological theory. */
    static const uint8_t FEATURES[26][5] = {
        /* a */ {0, 0, 1, 3, 1}, /* b */ {1, 0, 1, 0, 0}, /* c */ {4, 0, 0, 0, 0},
        /* d */ {2, 0, 1, 0, 0}, /* e */ {0, 0, 1, 2, 1}, /* f */ {3, 1, 0, 0, 0},
        /* g */ {4, 0, 1, 0, 0}, /* h */ {7, 1, 0, 0, 0}, /* i */ {0, 0, 1, 1, 1},
        /* j */ {5, 2, 1, 0, 0}, /* k */ {4, 0, 0, 0, 0}, /* l */ {2, 3, 1, 0, 0},
        /* m */ {1, 4, 1, 0, 0}, /* n */ {2, 4, 1, 0, 0}, /* o */ {0, 0, 1, 2, 1},
        /* p */ {1, 0, 0, 0, 0}, /* q */ {4, 0, 0, 0, 0}, /* r */ {5, 3, 1, 0, 0},
        /* s */ {2, 1, 0, 0, 0}, /* t */ {2, 0, 0, 0, 0}, /* u */ {0, 0, 1, 1, 1},
        /* v */ {3, 1, 1, 0, 0}, /* w */ {1, 5, 1, 0, 0}, /* x */ {2, 1, 0, 0, 0},
        /* y */ {5, 5, 1, 0, 0}, /* z */ {2, 1, 1, 0, 0},
    };

    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    if (len_a == 0 || len_b == 0) return 0.0f;

    /* Length ratio penalty — very different lengths are unlikely matches */
    float len_ratio = (float)(len_a < len_b ? len_a : len_b) /
                      (float)(len_a > len_b ? len_a : len_b);
    if (len_ratio < 0.5f) return 0.0f;

    /* Compare feature overlap using a sliding alignment.
     * For each letter in 'a', find the most similar letter in a window of 'b'. */
    float total_sim = 0.0f;
    uint32_t comparisons = 0;

    for (size_t i = 0; i < len_a; i++) {
        char ca = (char)tolower((unsigned char)a[i]);
        if (ca < 'a' || ca > 'z') continue;
        uint32_t fa = ca - 'a';

        float best_match = 0.0f;
        /* Search in a window around the proportional position in b */
        size_t center = (i * len_b) / len_a;
        size_t win_start = (center >= 2) ? center - 2 : 0;
        size_t win_end = (center + 3 < len_b) ? center + 3 : len_b;

        for (size_t j = win_start; j < win_end; j++) {
            char cb = (char)tolower((unsigned char)b[j]);
            if (cb < 'a' || cb > 'z') continue;
            uint32_t fb = cb - 'a';

            /* Count matching features */
            uint32_t matches = 0;
            for (int f = 0; f < 5; f++) {
                if (FEATURES[fa][f] == FEATURES[fb][f]) matches++;
            }
            float sim = (float)matches / 5.0f;
            if (sim > best_match) best_match = sim;
        }

        total_sim += best_match;
        comparisons++;
    }

    if (comparisons == 0) return 0.0f;
    return (total_sim / (float)comparisons) * len_ratio;
}

/**
 * Fuzzy character-set similarity using membership degree vectors.
 *
 * Models each word as a fuzzy set over the alphabet: each letter's membership
 * is its normalized frequency in the word. Then computes Jaccard-style
 * similarity: sum(min) / sum(max). Catches keyboard/visual typos:
 *   "calcuus" ↔ "calculus"  (missing letter)
 *   "teh" ↔ "the"  (transposition)
 *   "recieve" ↔ "receive"  (swapped letters)
 *
 * Returns similarity in [0, 1].
 */
static float fuzzy_charset_similarity(const char* a, const char* b) {
    float freq_a[26] = {0};
    float freq_b[26] = {0};
    uint32_t count_a = 0, count_b = 0;

    for (const char* p = a; *p; p++) {
        char c = (char)tolower((unsigned char)*p);
        if (c >= 'a' && c <= 'z') {
            freq_a[c - 'a'] += 1.0f;
            count_a++;
        }
    }
    for (const char* p = b; *p; p++) {
        char c = (char)tolower((unsigned char)*p);
        if (c >= 'a' && c <= 'z') {
            freq_b[c - 'a'] += 1.0f;
            count_b++;
        }
    }

    if (count_a == 0 || count_b == 0) return 0.0f;

    /* Normalize to membership degrees [0, 1] */
    for (int i = 0; i < 26; i++) {
        freq_a[i] /= (float)count_a;
        freq_b[i] /= (float)count_b;
    }

    /* Fuzzy Jaccard: sum(min(A,B)) / sum(max(A,B)) */
    float sum_min = 0.0f, sum_max = 0.0f;
    for (int i = 0; i < 26; i++) {
        sum_min += (freq_a[i] < freq_b[i]) ? freq_a[i] : freq_b[i];
        sum_max += (freq_a[i] > freq_b[i]) ? freq_a[i] : freq_b[i];
    }

    if (sum_max < 1e-8f) return 0.0f;
    float base_sim = sum_min / sum_max;

    /* Length penalty — character bag similarity is high for anagrams,
     * so penalize large length differences */
    float len_ratio = (float)(count_a < count_b ? count_a : count_b) /
                      (float)(count_a > count_b ? count_a : count_b);

    return base_sim * (0.5f + 0.5f * len_ratio);
}

/**
 * Find best fuzzy match in lexicon when exact lookup fails.
 *
 * Two-stage scoring:
 *   Stage 1: Phonological similarity (catches sound-alike typos)
 *   Stage 2: Fuzzy character-set similarity (catches visual/keyboard typos)
 * Takes the max of both scores. Accepts if above GL_FUZZY_THRESHOLD.
 *
 * Scans vocab_list linearly — O(vocab_count) but only called on unknown words,
 * and typical vocab is < 16K entries with short string comparisons.
 */
static const gl_lexicon_entry_t* lexicon_find_fuzzy(const grounded_language_t* gl,
                                                      const char* word) {
    char lower[GL_MAX_WORD_LEN];
    lowercase_word(word, lower, GL_MAX_WORD_LEN);

    size_t len = strlen(lower);
    if (len < GL_FUZZY_MIN_LEN) return NULL;

    const gl_lexicon_entry_t* best_entry = NULL;
    float best_score = GL_FUZZY_THRESHOLD;

    for (uint32_t i = 0; i < gl->vocab_count; i++) {
        const gl_lexicon_entry_t* candidate = gl->vocab_list[i];
        if (!candidate) continue;

        /* Quick length filter — skip if lengths differ by more than 40% */
        size_t clen = strlen(candidate->form);
        if (clen < GL_FUZZY_MIN_LEN) continue;
        float len_ratio = (float)(len < clen ? len : clen) /
                          (float)(len > clen ? len : clen);
        if (len_ratio < 0.6f) continue;

        /* Stage 1: Phonological similarity */
        float phon_score = phonological_similarity(lower, candidate->form);

        /* Stage 2: Fuzzy character-set similarity */
        float fuzzy_score = fuzzy_charset_similarity(lower, candidate->form);

        /* Sanity floor: the *weaker* signal must clear a small bar.
         * Phonological alone is too lenient (5-feature space) and will
         * "match" nonsense like 'zxqvkbjpfm' to 'surprising' at 0.7+.
         * Real typos clear both signals; nonsense doesn't. */
        float weaker = (phon_score < fuzzy_score) ? phon_score : fuzzy_score;
        if (weaker < GL_FUZZY_MIN_SIGNAL) continue;

        /* Take the better of the two approaches */
        float score = (phon_score > fuzzy_score) ? phon_score : fuzzy_score;

        if (score > best_score) {
            best_score = score;
            best_entry = candidate;
        }
    }

    if (best_entry) {
        LOG_DEBUG(LOG_MODULE, "Fuzzy match: '%s' -> '%s' (score=%.3f)",
                  lower, best_entry->form, best_score);
    }

    return best_entry;
}

/* External alias for the NLP frontend (sibling .c file). */
const gl_lexicon_entry_t* lexicon_find_fuzzy_external(const grounded_language_t* gl,
                                                        const char* word) {
    return lexicon_find_fuzzy(gl, word);
}

/** Mirror a lexicon word↔concept binding into the SNN language bridge.
 *
 * Without this, the bridge has zero spike→word bindings (verified live:
 * bridge_active_bindings = 0 at every checkpoint we've inspected). The
 * decode_spikes path always returns empty, produce always returns -1,
 * and respond falls through to "I don't know."
 *
 * Mapping:
 *   word_pop    = entry->form_hash % SNN_LANG_MAX_WORD_POPS    (16384)
 *   concept_pop = (concept_id mod 2^32) % SNN_LANG_MAX_CONCEPT_POPS  (4096)
 *
 * Hash collisions are accepted: at 30K vocab → 16K word_pops, ~50% of
 * pops hold one word and the rest share. Words sharing a pop become
 * synonyms-by-collision under decoding. Suboptimal but functional, and
 * vastly better than the all-zero baseline. The bridge's STDP layer
 * separates the colliders over training as the spike pre/post traces
 * diverge for differently-grounded words.
 *
 * D1 — Spike-driven STDP (2026-05-06):
 * Synthetic spike events at bind time were tried (commit 5d47666ae) but
 * caused SNN sparsity collapse: concept_pops[].activation has no decay
 * anywhere in the bridge. Each synthesized concept_spike incremented
 * activation by +1.0 forever. With 83K bulk-load + sidecar binds, the
 * bridge's attention-feedback path (generate_attention_feedback →
 * visual/audio attention_gains EMA) saw uniform-positive activations
 * across nearly all concept_pops, broadcast "pay attention to everything"
 * to the sensory bridges, and drove every neuron in the SNN to fire at
 * least once per window → sparsity dropped from 0.50 to 0.00 and stuck.
 *
 * Reverted to bind-only. STDP becomes a no-op again (deferred D1 — the
 * real fix is to wire SNN population spike events through the bridge,
 * not synthesize them at lexicon-bind time, AND/OR add decay to the
 * activation accumulators). Bindings still live; bridge production still
 * works; weights stay at the initial mirror-pushed values until the
 * proper SNN-spike wiring lands.
 */
static void mirror_binding_to_bridge(grounded_language_t* gl,
                                      const gl_lexicon_entry_t* entry,
                                      uint64_t concept_id,
                                      float strength) {
    if (!gl->snn_bridge || !entry) return;
    uint32_t word_pop = entry->form_hash % SNN_LANG_MAX_WORD_POPS;
    uint32_t concept_pop = (uint32_t)(concept_id % SNN_LANG_MAX_CONCEPT_POPS);
    /* register_word/register_concept are idempotent (overwrite slot). bind
     * requires both pops registered (it checks num_*_pops). */
    snn_language_bridge_register_word(gl->snn_bridge, word_pop, entry->form);
    snn_language_bridge_register_concept(gl->snn_bridge, concept_pop, concept_id);
    snn_language_bridge_bind(gl->snn_bridge, concept_pop, word_pop, strength);
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
            mirror_binding_to_bridge(gl, entry, concept_id, entry->bindings[i].strength);
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
    mirror_binding_to_bridge(gl, entry, concept_id, strength);
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

/** Seed conceptual vocabulary — real words Athena needs to communicate */
static void seed_conceptual_words(grounded_language_t* gl) {
    /* Cognitive/conversational nouns */
    static const char* nouns[] = {
        "thought", "idea", "question", "answer", "meaning", "pattern",
        "knowledge", "understanding", "experience", "memory", "feeling",
        "reason", "logic", "concept", "problem", "solution", "hypothesis",
        "evidence", "truth", "belief", "consciousness", "awareness",
        "learning", "thinking", "mind", "brain", "intelligence",
        "language", "word", "sentence", "conversation", "response",
        "science", "mathematics", "physics", "biology", "chemistry",
        "philosophy", "ethics", "morality", "justice", "beauty",
        "nature", "universe", "energy", "matter", "space", "time",
        "life", "death", "growth", "change", "process", "system",
        "structure", "function", "purpose", "goal", "action",
        "observation", "prediction", "surprise", "curiosity",
        "emotion", "confidence", "uncertainty", "doubt", "wonder",
        "connection", "relationship", "similarity", "difference",
        NULL
    };
    for (int i = 0; nouns[i]; i++) {
        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, nouns[i]);
        if (entry) {
            entry->learned_class = GL_CLASS_NOUN;
            entry->class_confidence = 0.7f;
            entry->frequency = 20;
        }
    }

    /* Cognitive/conversational verbs */
    static const char* verbs[] = {
        "think", "know", "understand", "learn", "remember", "believe",
        "feel", "sense", "perceive", "observe", "notice", "recognize",
        "mean", "explain", "describe", "say", "tell", "ask", "answer",
        "consider", "wonder", "imagine", "predict", "expect", "surprise",
        "analyze", "reason", "compare", "connect", "relate", "combine",
        "create", "discover", "find", "search", "explore", "investigate",
        "help", "try", "want", "need", "like", "enjoy", "prefer",
        "agree", "disagree", "doubt", "trust", "hope", "fear",
        "change", "grow", "develop", "evolve", "improve", "adapt",
        "exist", "become", "seem", "appear", "happen", "occur",
        NULL
    };
    for (int i = 0; verbs[i]; i++) {
        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, verbs[i]);
        if (entry) {
            entry->learned_class = GL_CLASS_VERB;
            entry->class_confidence = 0.7f;
            entry->frequency = 20;
        }
    }

    /* Cognitive/conversational adjectives */
    static const char* adjectives[] = {
        "interesting", "important", "complex", "simple", "clear",
        "uncertain", "confident", "curious", "surprising", "familiar",
        "new", "different", "similar", "related", "relevant",
        "true", "false", "possible", "likely", "unlikely",
        "good", "bad", "right", "wrong", "useful", "helpful",
        "beautiful", "abstract", "concrete", "fundamental",
        "logical", "emotional", "rational", "intuitive", "creative",
        NULL
    };
    for (int i = 0; adjectives[i]; i++) {
        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, adjectives[i]);
        if (entry) {
            entry->learned_class = GL_CLASS_ADJECTIVE;
            entry->class_confidence = 0.7f;
            entry->frequency = 15;
        }
    }

    /* Adverbs */
    static const char* adverbs[] = {
        "deeply", "carefully", "clearly", "slowly", "quickly",
        "perhaps", "probably", "certainly", "truly", "really",
        "together", "already", "still", "yet", "always", "never",
        NULL
    };
    for (int i = 0; adverbs[i]; i++) {
        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, adverbs[i]);
        if (entry) {
            entry->learned_class = GL_CLASS_ADVERB;
            entry->class_confidence = 0.7f;
            entry->frequency = 10;
        }
    }
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

/* Score one word against a target vector — DRY extraction so the loop
 * body in find_words_near_vector stays readable and the same scoring
 * function can be reused by tests. Returns the combined
 * distributional + grounded score, or 0.0f if the entry should be
 * skipped. The caller is responsible for class filtering. */
static float score_word_against_vector(const grounded_language_t* gl,
                                       const gl_lexicon_entry_t* entry,
                                       const float* target, uint32_t dim) {
    if (!entry || !entry->context_initialized) return 0.0f;
    uint32_t cmp_dim = (dim < gl->semantic_dim) ? dim : gl->semantic_dim;
    float sim = cosine_similarity(target, entry->context_vector, cmp_dim);

    float concept_sim = 0.0f;
    for (uint32_t b = 0; b < entry->binding_count; b++) {
        const float* cf = get_concept_features(gl, entry->bindings[b].concept_id);
        if (cf) {
            float cs = cosine_similarity(target, cf, cmp_dim);
            cs *= entry->bindings[b].strength;
            if (cs > concept_sim) concept_sim = cs;
        }
    }
    return 0.4f * sim + 0.6f * concept_sim;
}

/** Find words closest to a semantic vector.
 *
 * Collapse guard (2026-05): when the lexicon hasn't accumulated enough
 * grounded bindings, score_word_against_vector() returns near-uniform
 * tiny values across all candidates. The original deterministic top-K
 * selection then pinned the same handful of words for every prompt,
 * which is the user-visible "the awareness controlled" mode collapse.
 *
 * Strategy: keep the deterministic top-K when any candidate scores
 * meaningfully (> GL_DIVERSITY_MIN_TOPSCORE). When the best score is
 * below that floor, treat the result as "no signal" and shuffle the
 * insertion order so produce() at least gets *different* words across
 * prompts (degenerate-but-diverse beats degenerate-and-stuck).
 */
#define GL_DIVERSITY_MIN_TOPSCORE 0.05f

static uint32_t find_words_near_vector(const grounded_language_t* gl,
                                        const float* target, uint32_t dim,
                                        gl_word_class_t required_class,
                                        const char** out_words, float* out_scores,
                                        uint32_t max_words) {
    uint32_t count = 0;
    float best_seen = 0.0f;

    for (uint32_t w = 0; w < gl->vocab_count && count < max_words; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry || !entry->context_initialized) continue;
        if (required_class != GL_CLASS_UNKNOWN && entry->learned_class != required_class) {
            continue;
        }
        float score = score_word_against_vector(gl, entry, target, dim);
        if (score > best_seen) best_seen = score;

        /* Insert into sorted output (simple insertion sort) */
        uint32_t insert_pos = count;
        for (uint32_t j = 0; j < count; j++) {
            if (score > out_scores[j]) {
                insert_pos = j;
                break;
            }
        }
        if (insert_pos < max_words) {
            if (count < max_words) count++;
            for (uint32_t j = count - 1; j > insert_pos; j--) {
                out_words[j] = out_words[j - 1];
                out_scores[j] = out_scores[j - 1];
            }
            out_words[insert_pos] = entry->form;
            out_scores[insert_pos] = score;
        }
    }

    /* Diversity injection: if no candidate scored above the meaningful
     * floor, the top-K is degenerate. Shuffle the result so callers
     * don't always see the same first word. We use a hash of the
     * target's first component as the seed so identical inputs still
     * map to identical orderings (deterministic, but per-input). */
    if (best_seen < GL_DIVERSITY_MIN_TOPSCORE && count > 1) {
        uint32_t seed = 0x9E3779B9u;
        for (uint32_t i = 0; i < dim && i < 8; i++) {
            uint32_t bits;
            memcpy(&bits, &target[i], sizeof(bits));
            seed ^= bits + (seed << 6) + (seed >> 2);
        }
        /* Fisher-Yates shuffle with deterministic LCG seeded from input. */
        uint32_t rng = (seed == 0) ? 1u : seed;
        for (uint32_t i = count - 1; i > 0; i--) {
            rng = rng * 1664525u + 1013904223u;
            uint32_t j = rng % (i + 1);
            const char* tw = out_words[i]; out_words[i] = out_words[j]; out_words[j] = tw;
            float ts = out_scores[i];      out_scores[i] = out_scores[j]; out_scores[j] = ts;
        }
    }

    return count;
}

/*=============================================================================
 * Lifecycle Implementation
 *===========================================================================*/

grounded_language_t* grounded_language_create(uint32_t semantic_dim, void* semantic_memory) {
    grounded_language_t* gl = (grounded_language_t*)nimcp_calloc(1, sizeof(grounded_language_t));
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_create: failed to allocate grounded_language_t");
        return NULL;
    }

    gl->semantic_dim = (semantic_dim > 0) ? semantic_dim : GL_SEMANTIC_DIM;
    gl->semantic_memory = (semantic_memory_system_t*)semantic_memory;

    /* Hash table: 2x max vocab for low collision rate */
    gl->lexicon_size = GL_MAX_VOCAB * 2;
    gl->lexicon = (gl_lexicon_entry_t**)nimcp_calloc(gl->lexicon_size, sizeof(gl_lexicon_entry_t*));
    gl->vocab_list = (gl_lexicon_entry_t**)nimcp_calloc(GL_MAX_VOCAB, sizeof(gl_lexicon_entry_t*));
    if (!gl->lexicon || !gl->vocab_list) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_create: failed to allocate lexicon tables");
        grounded_language_destroy(gl);
        return NULL;
    }

    /* Context */
    gl->context.context_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!gl->context.context_vector) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_create: failed to allocate context vector");
        grounded_language_destroy(gl);
        return NULL;
    }

    /* Compositional phrases (#9) */
    gl->phrases = (gl_phrase_t*)nimcp_calloc(GL_MAX_PHRASES, sizeof(gl_phrase_t));
    if (!gl->phrases) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_create: failed to allocate phrase table");
        grounded_language_destroy(gl);
        return NULL;
    }
    gl->phrase_count = 0;

    /* Learning parameters */
    gl->hebbian_lr = GL_HEBBIAN_LR_DEFAULT;
    gl->decay_rate = GL_DECAY_RATE_DEFAULT;

    /* Tier-2 defaults: negation ON (cheap, biologically realistic),
     * sense disambiguation OFF (opt-in — preserves legacy activation
     * profile bit-for-bit on resume of a pre-Tier-2 brain). Discourse
     * starts empty with capacity = full GL_DISCOURSE_MAX_TURNS. */
    gl->enable_negation_inversion   = true;
    gl->enable_sense_disambiguation = false;
    gl->discourse.head     = 0;
    gl->discourse.count    = 0;
    gl->discourse.capacity = GL_DISCOURSE_MAX_TURNS;

    /* RNG */
    gl->rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEULL;

    /* Seed with function and conceptual words. Syntactic structure is
     * learned emergently via SNN bridge plasticity, not seeded as templates. */
    seed_function_words(gl);
    seed_conceptual_words(gl);

    /* Register with BBB */
    bbb_register_module("grounded_language", BBB_MODULE_TYPE_COGNITIVE);

    LOG_INFO(LOG_MODULE, "Grounded language system created (dim=%u, vocab=%u words)",
             gl->semantic_dim, gl->vocab_count);

    return gl;
}

/* Walkthrough round 3 fix: forward decls for static helpers defined later
 * in this file but called from earlier functions. */
static void gl_detach_spectrum_for_destroy(grounded_language_t* gl);
static void gl_next_token_cold_start_skips_inc(const char* prev, const char* next);
/* Tier-1 #2: anaphora-resolver forward decls (defined later, near the
 * spectrum side-map). */
static void gl_detach_anaphora_for_destroy(grounded_language_t* gl);
static uint32_t anaphora_run_pass(grounded_language_t* gl,
                                    char** words,
                                    uint32_t word_count,
                                    uint64_t* activated_concepts,
                                    float* activation_levels,
                                    uint32_t* concept_count_io);

void grounded_language_destroy(grounded_language_t* gl) {
    if (!gl) return;

    /* Walkthrough round 3 fix: clear our entry from the static spectrum map
     * BEFORE freeing the gl, so the next gl_get_attached_spectrum cannot
     * return a spectrum bound to this freed pointer (UAF if the OS reuses
     * this address for a fresh grounded_language). */
    gl_detach_spectrum_for_destroy(gl);
    /* Tier-1 #2: same UAF guard for the anaphora side-map. */
    gl_detach_anaphora_for_destroy(gl);

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
    nimcp_free(gl->context.context_vector);

    /* Phrases (#9) — each entry's semantic_vec is gl-owned. */
    if (gl->phrases) {
        for (uint32_t p = 0; p < gl->phrase_count; p++) {
            nimcp_free(gl->phrases[p].semantic_vec);
        }
        nimcp_free(gl->phrases);
    }

    /* Tier-2 #7 — free per-turn semantic vectors. The turns array
     * itself is inline in the struct, so only the heap-owned vectors
     * need releasing. */
    for (uint8_t t = 0; t < GL_DISCOURSE_MAX_TURNS; t++) {
        nimcp_free(gl->discourse.turns[t].semantic_vector);
    }

    nimcp_free(gl);
}

void grounded_language_set_semantic_memory(grounded_language_t* gl, void* semantic_memory) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_set_semantic_memory: NULL gl");
        return;
    }
    gl->semantic_memory = (semantic_memory_system_t*)semantic_memory;
}

/*=============================================================================
 * Comprehension
 *===========================================================================*/

/* Tier-2 forward decls — implementations live just below
 * gl_comprehension_result_cleanup (kept colocated with the rest of the
 * Tier-2 helpers). Forward-declared here so grounded_language_comprehend
 * can call them without reordering the file. */
static bool is_function_word(const char* w);
static bool is_negation_cue(const char* w);

int grounded_language_comprehend(grounded_language_t* gl, const char* text,
                                  gl_comprehension_result_t* result) {
    if (!gl || !text || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_comprehend: NULL parameter (gl=%p, text=%p, result=%p)",
            (void*)gl, (void*)text, (void*)result);
        return -1;
    }
    memset(result, 0, sizeof(*result));

    /* Tokenize */
    size_t text_len = strlen(text);
    char* buf = (char*)nimcp_malloc(text_len + 1);
    if (!buf) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_comprehend: failed to allocate text buffer (%zu bytes)", text_len + 1);
        return -1;
    }
    memcpy(buf, text, text_len + 1);

    char* words[GL_MAX_PRODUCTION_WORDS];
    uint32_t word_count = tokenize_text(buf, words, GL_MAX_PRODUCTION_WORDS);

    /* Lower-case in-place so cue / function-word checks match the
     * lexicon's stored forms. The tokenizer emitted pointers into buf
     * — overwriting them here is safe (we never need the original
     * mixed-case form again in this scope). */
    for (uint32_t w = 0; w < word_count; w++) {
        for (char* p = words[w]; *p; p++) {
            *p = (char)tolower((unsigned char)*p);
        }
    }

    /* Tier-2 #3 negation: scan for cues and mark the next non-function
     * content word within GL_NEGATION_WINDOW for activation-sign flip.
     * `negate_word[w]` = true means activations contributed by this
     * word (and folded into result->activation_levels) get negated. */
    bool negate_word[GL_MAX_PRODUCTION_WORDS];
    memset(negate_word, 0, sizeof(negate_word));
    bool any_negated = false;
    if (gl->enable_negation_inversion) {
        for (uint32_t i = 0; i < word_count; i++) {
            if (!is_negation_cue(words[i])) continue;
            uint32_t look_end = i + 1 + GL_NEGATION_WINDOW;
            if (look_end > word_count) look_end = word_count;
            for (uint32_t j = i + 1; j < look_end; j++) {
                if (is_function_word(words[j])) continue;
                if (is_negation_cue(words[j])) continue; /* skip stacked "not no" */
                negate_word[j] = true;
                any_negated = true;
                break;
            }
        }
    }

    /* Allocate result arrays */
    result->activated_concepts = (uint64_t*)nimcp_calloc(GL_MAX_ACTIVE_CONCEPTS, sizeof(uint64_t));
    result->activation_levels = (float*)nimcp_calloc(GL_MAX_ACTIVE_CONCEPTS, sizeof(float));
    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!result->activated_concepts || !result->activation_levels || !result->semantic_vector) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_comprehend: failed to allocate result arrays");
        nimcp_free(buf);
        gl_comprehension_result_cleanup(result);
        return -1;
    }

    /* Process each word */
    float total_activation = 0.0f;
    uint32_t known_words = 0;
    uint32_t concept_count = 0;
    bool sense_resolved_this_pass = false;

    /* #10 Active-learning curriculum signal — track the first word that
     * either misses the lexicon or has weak bindings, so we can fire a
     * single NEEDS_GROUNDING event below pointed at the most-needy
     * token (rather than per-word spam). */
    const char* needy_word = NULL;
    float       needy_word_conf = 1.0f;

    /* NLP frontend: morphological normalization between exact and fuzzy.
     * gl_nlp_lookup_chain walks exact → morph-strip → fuzzy in order. */
    extern const gl_lexicon_entry_t* gl_nlp_lookup_chain(grounded_language_t*,
                                                           const char*);
    extern int gl_nlp_subword_lookup(grounded_language_t*, const char*, float*);

    uint32_t subword_hits = 0;

    for (uint32_t w = 0; w < word_count; w++) {
        const gl_lexicon_entry_t* entry = gl_nlp_lookup_chain(gl, words[w]);
        if (!entry) {
            /* Last resort: BPE subword fallback when a tokenizer is
             * connected. Doesn't yield an entry — just bumps confidence
             * and (if embeddings are wired) blends subword vectors into
             * the semantic_vector below. */
            if (gl_nlp_subword_lookup(gl, words[w],
                                       result->semantic_vector) == 0) {
                subword_hits++;
            }
            /* Truly unknown word — candidate for active-learning hook. */
            if (!needy_word) needy_word = words[w];
            needy_word_conf = 0.0f;
            continue;
        }

        known_words++;

        /* #10 Track weakest known word for the active-learning hook. */
        float top_strength = 0.0f;
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            if (entry->bindings[b].strength > top_strength)
                top_strength = entry->bindings[b].strength;
        }
        if (top_strength < needy_word_conf) {
            needy_word = words[w];
            needy_word_conf = top_strength;
        }

        /* Tier-2 #6 sense disambiguation. Default OFF — when enabled
         * and the entry has > 1 binding, pick the binding that best
         * aligns with the current running context vector and damp the
         * off-sense bindings to GL_SENSE_OFF_DAMP. The chosen binding
         * keeps weight 1.0 so its strength × weight matches legacy
         * single-binding behaviour. The decision uses gl->context.
         * context_vector (the running discourse blend) as the intent
         * proxy; that vector is non-zero whenever any prior comprehend
         * or push_turn has populated it. With an all-zero context
         * (cold-start), disambiguate_sense returns 0 and the chosen
         * binding is simply the first one — equivalent to legacy. */
        uint32_t best_sense = 0;
        bool     do_sense_weighting = false;
        if (gl->enable_sense_disambiguation && entry->binding_count > 1) {
            best_sense = grounded_language_disambiguate_sense(gl, entry,
                                                                gl->context.context_vector);
            do_sense_weighting = true;
            sense_resolved_this_pass = true;
        }

        /* Activate all bound concepts */
        const float polarity = negate_word[w] ? -1.0f : 1.0f;
        for (uint32_t b = 0; b < entry->binding_count && concept_count < GL_MAX_ACTIVE_CONCEPTS; b++) {
            const gl_word_binding_t* binding = &entry->bindings[b];
            if (binding->strength < GL_ASSOC_PRUNE_THRESHOLD) continue;

            /* Sense weight: 1.0 for the chosen binding when WSD is on,
             * GL_SENSE_OFF_DAMP for the others; 1.0 unconditionally
             * when WSD is off (legacy). */
            float sense_w = 1.0f;
            if (do_sense_weighting) {
                sense_w = (b == best_sense) ? 1.0f : GL_SENSE_OFF_DAMP;
            }
            float effective = binding->strength * sense_w;

            /* Check if concept already activated */
            bool found = false;
            for (uint32_t c = 0; c < concept_count; c++) {
                if (result->activated_concepts[c] == binding->concept_id) {
                    result->activation_levels[c] += polarity * effective;
                    found = true;
                    break;
                }
            }

            if (!found) {
                result->activated_concepts[concept_count] = binding->concept_id;
                result->activation_levels[concept_count] = polarity * effective;
                concept_count++;
            }

            total_activation += effective;

            /* Add concept features to semantic vector. Negation does
             * NOT flip the semantic vector contribution — keeping the
             * topical embedding intact while only the polarity / sign
             * of activation_levels carries the negation signal lets
             * downstream pipelines (cosine similarity, network
             * broadcast) still use the vector as a topic descriptor.
             * The activation list is the canonical "what was meant"
             * channel and that's where the sign flip lives. */
            const float* cf = get_concept_features(gl, binding->concept_id);
            if (cf) {
                float weight = effective / (float)word_count;
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

        /* NLP: if a word-embedding layer is connected and the dim
         * matches semantic_dim, fold the per-word embedding into the
         * semantic vector. Caller-owned ctx + word→id callback so this
         * file doesn't need to know the tokenizer's vocab. */
        extern int gl_nlp_embedding_lookup(grounded_language_t*, const char*, float*);
        float emb_buf[1024];
        if (gl->semantic_dim <= 1024 &&
            gl_nlp_embedding_lookup(gl, words[w], emb_buf) == 0) {
            float emb_weight = 0.2f / (float)word_count;
            for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                result->semantic_vector[d] += emb_buf[d] * emb_weight;
            }
        }
    }

    /* Tier-1 #2: rule-based anaphora resolution. Disabled by default; opt-in
     * via grounded_language_set_anaphora_enabled(gl,true). Walks the same
     * token list (words[]) we just consumed, classifies each token as
     * pronoun or content noun, pushes referents onto a per-gl ring, and
     * folds the matched referent's lexicon-entry bindings into the
     * activated_concepts list at half-strength. The pass is a no-op when
     * disabled — production traffic sees zero overhead. */
    (void)anaphora_run_pass(gl, words, word_count,
                              result->activated_concepts,
                              result->activation_levels,
                              &concept_count);

    /* Engram integration (read-only). Snapshot the pre-engram activated
     * set so:
     *   (a) we ENCODE only the comprehension-derived activations into a
     *       new engram trace — recall blend below does NOT feed back into
     *       what gets stored, preserving the read-only contract.
     *   (b) we use the snapshot as the RECALL cue, then blend recalled
     *       neurons (mapped 1:1 to concept_ids by lo32 hash equivalence)
     *       into the caller-visible activated_concepts at half-weight.
     * Both stages no-op when no engram_system is attached or the flag
     * is off. */
    if (gl->engram_enabled && gl->engram_system && concept_count > 0) {
        /* Truncate concept_id u64 → neuron_id u32 (lower 32 bits).
         * Concept IDs are already derived from string hashes; the lo32
         * is a uniform 32-bit projection. We borrow up to 256 ids — the
         * engram API caps neuron_count at ENGRAM_MAX_NEURONS=256. */
        uint32_t cue_count = (concept_count > 256) ? 256 : concept_count;
        uint32_t cue_neurons[256];
        float    cue_activations[256];
        for (uint32_t i = 0; i < cue_count; i++) {
            cue_neurons[i] = (uint32_t)(result->activated_concepts[i] & 0xFFFFFFFFu);
            cue_activations[i] = result->activation_levels[i];
            /* engram_encode expects nonnegative activations — fold sign
             * into magnitude here so negation-marked words still encode. */
            if (cue_activations[i] < 0.0f) cue_activations[i] = -cue_activations[i];
        }

        /* EN-3 — encode the snapshot as an EPISODIC trace. Neutral
         * emotion (0,0); future task can wire emotional context if
         * available. The engram_id is intentionally discarded — for
         * read-only mode the trace lives in the engram store and is
         * looked up later via cue, not by id. */
        emotional_tag_t emo = {0};
        (void)engram_encode((engram_system_t*)gl->engram_system,
                             cue_neurons, cue_activations, cue_count,
                             MEMORY_TYPE_EPISODIC, emo);
        gl->stats.engram_encodes++;

        /* EN-4 — recall: probe with the same cue, blend recalled neurons
         * into activated_concepts at half-weight. Recall returns the
         * neuron_ids of the matched engram (same lo32 hashes) — we map
         * them back into concept activations by extending the array.
         * Threshold: ENGRAM_RECALL_THRESHOLD (0.4); we accept anything
         * the engram module deems confident enough. */
        uint32_t recalled[256];
        float    recalled_acts[256];
        float    confidence = 0.0f;
        uint64_t hit = engram_recall((engram_system_t*)gl->engram_system,
                                      cue_neurons, cue_count,
                                      recalled, recalled_acts, 256,
                                      &confidence);
        if (hit != 0 && confidence >= ENGRAM_RECALL_THRESHOLD) {
            gl->stats.engram_recalls++;
            for (uint32_t r = 0;
                 r < 256 && concept_count < GL_MAX_ACTIVE_CONCEPTS;
                 r++) {
                if (recalled_acts[r] <= 0.0f) continue;
                /* Half-weight blend, matches anaphora's pattern. The
                 * lo32 → concept_id map is lossy; we treat recalled[r]
                 * as a concept_id directly (all our concept_ids fit in
                 * 32 bits in practice for the brain's current scale). */
                uint64_t cid = (uint64_t)recalled[r];
                bool merged = false;
                for (uint32_t c = 0; c < concept_count; c++) {
                    if (result->activated_concepts[c] == cid) {
                        result->activation_levels[c] += 0.5f * recalled_acts[r];
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    result->activated_concepts[concept_count] = cid;
                    result->activation_levels[concept_count]   = 0.5f * recalled_acts[r];
                    concept_count++;
                }
            }
        }
    }

    result->concept_count = concept_count;
    /* Subword-recognized words count as half-known: the BPE tokenizer
     * found valid subword tokens for them but no full lexicon entry. */
    float effective_known = (float)known_words + 0.5f * (float)subword_hits;
    if (effective_known > (float)word_count) effective_known = (float)word_count;
    result->comprehension_confidence = (word_count > 0) ?
        effective_known / (float)word_count : 0.0f;
    result->novelty = (word_count > 0) ?
        1.0f - effective_known / (float)word_count : 1.0f;

    /* Normalize semantic vector */
    normalize_vector(result->semantic_vector, gl->semantic_dim);

    /* Walkthrough round 3 fix: detect all-zero semantic_vector (every word
     * was OOV with no fallback signal). Without this guard, the context
     * update below blends 0.7 zeros into the running context every time,
     * decaying it toward zero and silently degrading subsequent
     * comprehensions. Skip the blend on zero-vec so the previous context
     * is preserved verbatim. */
    float vec_l2 = 0.0f;
    for (uint32_t d = 0; d < gl->semantic_dim; d++) {
        vec_l2 += result->semantic_vector[d] * result->semantic_vector[d];
    }
    bool semantic_vec_is_zero = (vec_l2 < 1e-12f);

    /* Update context — skip if vector is zero (preserve prior context).
     *
     * Tier-2 #7: auto-push this comprehension as a discourse turn.
     * push_turn rebuilds gl->context.context_vector from the buffer's
     * recency-weighted blend, which supersedes the legacy 0.7 blend
     * but produces a comparable smoothed vector (geometric decay 0.6
     * across up to capacity turns vs. the legacy 70/30 single step).
     * Rebuilding from the buffer keeps the canonical context
     * consistent with the per-turn breakdown — without this, the
     * single rolling vector and the buffer would drift apart and any
     * future coherence check would have to choose one as canonical. */
    if (!semantic_vec_is_zero) {
        grounded_language_push_turn(gl, result->semantic_vector,
                                      gl->semantic_dim, word_count, true);
    }
    gl->context.words_in_context += word_count;

    /* Store recent concepts */
    gl->context.last_concept_count = 0;
    for (uint32_t c = 0; c < concept_count && c < 16; c++) {
        gl->context.last_concepts[c] = result->activated_concepts[c];
        gl->context.last_concept_count++;
    }

    gl->stats.total_comprehensions++;

    /* Tier-2 #3 / #6 telemetry — bump only on actual events. */
    if (gl->enable_negation_inversion && any_negated) {
        gl->stats.negation_events++;
    }
    if (sense_resolved_this_pass) {
        gl->stats.sense_resolutions++;
    }

    /* Dual-path: also feed through SNN bridge for spike-level comprehension */
    if (gl->snn_bridge) {
        float snn_concepts[512];
        uint32_t snn_activated = 0;
        float snn_conf = 0.0f;
        snn_language_bridge_comprehend(gl->snn_bridge, text,
                                       snn_concepts, 512, &snn_activated, &snn_conf);
        /* Blend spike-level confidence into result */
        float blend = snn_language_bridge_get_blend(gl->snn_bridge);
        result->comprehension_confidence =
            result->comprehension_confidence * (1.0f - blend) + snn_conf * blend;
    }

    /* Cortex modulation: scale comprehension confidence when sensory
     * cortexes are currently active. Caps at 1.0. No-op when no cortex
     * is connected (all scalars stay 0). */
    extern int grounded_language_get_cortex_modulation(grounded_language_t*,
                                                        gl_cortex_modulation_t*);
    gl_cortex_modulation_t mod = {0};
    if (grounded_language_get_cortex_modulation(gl, &mod) == 0) {
        float peak = mod.visual_activity;
        if (mod.audio_salience > peak)    peak = mod.audio_salience;
        if (mod.speech_confidence > peak) peak = mod.speech_confidence;
        if (peak < 0.0f) peak = 0.0f;
        if (peak > 1.0f) peak = 1.0f;
        result->comprehension_confidence *= (1.0f + 0.2f * peak);
        if (result->comprehension_confidence > 1.0f)
            result->comprehension_confidence = 1.0f;
    }

    /* Per-network broadcast: feed the final semantic_vector to LNN /
     * CNN / FNO / ANN if attached, then apply their averaged response
     * magnitude as a confidence boost up to +15%. No-op (factor=1.0)
     * when no networks are wired. */
    extern int grounded_language_broadcast_to_networks(grounded_language_t*,
                                                         const float*, uint32_t);
    extern float gl_network_modulation_factor(grounded_language_t*, float);
    grounded_language_broadcast_to_networks(gl, result->semantic_vector,
                                              gl->semantic_dim);
    result->comprehension_confidence *= gl_network_modulation_factor(gl, 0.15f);
    if (result->comprehension_confidence > 1.0f)
        result->comprehension_confidence = 1.0f;

    /* Fire COMPREHENDED event on the cognitive bus. */
    extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
    gl_event_t bus_ev = {0};
    bus_ev.type         = GL_EVENT_COMPREHENDED;
    bus_ev.text         = text;
    bus_ev.semantic_vec = result->semantic_vector;
    bus_ev.confidence   = result->comprehension_confidence;
    gl_fire_event(gl, &bus_ev);

    /* #10 Active-learning curriculum signal. If overall comprehension
     * confidence is low or the weakest word's strength is below the
     * threshold, fire NEEDS_GROUNDING with the targeted word so
     * curriculum modules can prioritize it for the next exposure. */
    if (needy_word &&
        (result->comprehension_confidence < GL_LOW_CONFIDENCE_THRESHOLD ||
         needy_word_conf < GL_LOW_CONFIDENCE_THRESHOLD)) {
        gl_event_t need_ev = {0};
        need_ev.type        = GL_EVENT_NEEDS_GROUNDING;
        need_ev.word        = needy_word;
        need_ev.text        = text;
        need_ev.confidence  = needy_word_conf;
        gl_fire_event(gl, &need_ev);
        gl->stats.total_needs_grounding++;
    }

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
 * Tier-2 #3 — Negation polarity helpers
 *
 * Implementation lives next to comprehend so the lookahead window
 * constants (GL_NEGATION_WINDOW), the function-word filter, and the
 * cue table all stay in one place. Cues are matched on the lower-cased
 * token; the contraction list catches both the standalone form
 * ("don't") and the n't-suffix-on-prior-token pattern handled by
 * is_negation_cue() returning true for either.
 *===========================================================================*/

static bool is_function_word(const char* w) {
    /* Cheap linear scan — FUNCTION_WORDS is ~70 entries and only walked
     * during negation lookahead (not the binding loop). */
    for (int i = 0; FUNCTION_WORDS[i]; i++) {
        if (strcmp(w, FUNCTION_WORDS[i]) == 0) return true;
    }
    return false;
}

static bool is_negation_cue(const char* w) {
    if (!w || !*w) return false;
    /* Bare cues. */
    static const char* CUES[] = {
        "not", "no", "never",
        "don't", "doesn't", "didn't",
        "won't", "can't", "cannot",
        "isn't", "wasn't", "aren't", "weren't",
        "haven't", "hasn't", "hadn't",
        "shouldn't", "wouldn't", "couldn't",
        NULL
    };
    for (int i = 0; CUES[i]; i++) {
        if (strcmp(w, CUES[i]) == 0) return true;
    }
    /* Suffix "n't" that survived the tokenizer (e.g. "don" / "t" got
     * glued back via the apostrophe). The tokenizer keeps apostrophes
     * inside words so most contractions arrive whole, but a defensive
     * suffix check costs almost nothing and catches "shan't" / dialect
     * forms that aren't in the table above. */
    size_t n = strlen(w);
    if (n >= 3 && strcmp(w + n - 3, "n't") == 0) return true;
    return false;
}

/*=============================================================================
 * Tier-2 #6 — Sense-disambiguation helper
 *
 * Picks the binding whose concept-features cosine-best aligns with the
 * supplied intent vector. Falls back to index 0 on missing features /
 * empty intents — deterministic so test assertions stay tight.
 *===========================================================================*/

uint32_t grounded_language_disambiguate_sense(
    const grounded_language_t* gl,
    const gl_lexicon_entry_t* entry,
    const float* intent_vec)
{
    if (!gl || !entry || !intent_vec) return 0;
    if (entry->binding_count == 0) return 0;
    if (entry->binding_count == 1) return 0;

    uint32_t best = 0;
    float    best_score = -2.0f; /* cosine ∈ [-1, 1] — sentinel below the floor */
    bool     any_scored = false;

    for (uint32_t b = 0; b < entry->binding_count; b++) {
        const float* cf = get_concept_features(gl, entry->bindings[b].concept_id);
        if (!cf) continue;
        float score = cosine_similarity(intent_vec, cf, gl->semantic_dim);
        any_scored = true;
        if (score > best_score) {
            best_score = score;
            best = b;
        }
    }

    /* No binding had concept features → fall back to first binding. */
    return any_scored ? best : 0;
}

/*=============================================================================
 * Tier-2 #3 / #6 — config setters/getters
 *===========================================================================*/

void grounded_language_set_negation_enabled(grounded_language_t* gl, bool enabled) {
    if (!gl) return;
    gl->enable_negation_inversion = enabled;
}

bool grounded_language_get_negation_enabled(const grounded_language_t* gl) {
    if (!gl) return false;
    return gl->enable_negation_inversion;
}

void grounded_language_set_sense_disambiguation_enabled(grounded_language_t* gl,
                                                         bool enabled) {
    if (!gl) return;
    gl->enable_sense_disambiguation = enabled;
}

bool grounded_language_get_sense_disambiguation_enabled(const grounded_language_t* gl) {
    if (!gl) return false;
    return gl->enable_sense_disambiguation;
}

/*=============================================================================
 * Tier-2 #7 — Multi-turn discourse buffer
 *
 * Implementation: small ring buffer of length capacity ≤ MAX_TURNS.
 * push_turn copies the supplied vector (or stores NULL for empty
 * placeholders); when the buffer is full the oldest slot is reclaimed.
 * The legacy gl->context.context_vector is recomputed every push as a
 * recency-weighted blend across the live turns.
 *===========================================================================*/

/* Map a logical position [0, count) where 0 = oldest, count-1 = newest
 * into the ring index. When count < capacity the buffer hasn't wrapped
 * and turns sit in [0, count) with head == 0; otherwise the oldest is
 * at `head`. */
static inline uint8_t discourse_idx(const gl_discourse_state_t* d, uint8_t pos) {
    return (uint8_t)((d->head + pos) % (d->capacity ? d->capacity : 1));
}

/* Recompute gl->context.context_vector as a recency-weighted blend of
 * the live discourse turns. Newest turn gets the largest weight; older
 * turns decay geometrically by 0.6 per step. Skips empty placeholder
 * turns (NULL semantic_vector). After the blend the vector is
 * normalized (unit L2) so downstream cosine consumers see a stable
 * scale. Called by push_turn; safe to call when count == 0 (no-op,
 * leaves the prior context untouched). */
static void discourse_rebuild_context_blend(grounded_language_t* gl) {
    if (!gl || gl->discourse.count == 0) return;
    if (!gl->context.context_vector) return;

    float* out = gl->context.context_vector;
    memset(out, 0, gl->semantic_dim * sizeof(float));

    /* Weights from oldest → newest: 0.6^(count-1-i). */
    float total_weight = 0.0f;
    for (uint8_t i = 0; i < gl->discourse.count; i++) {
        const gl_discourse_turn_t* t =
            &gl->discourse.turns[discourse_idx(&gl->discourse, i)];
        if (!t->semantic_vector) continue;
        float w = 1.0f;
        uint8_t depth = (uint8_t)(gl->discourse.count - 1 - i);
        for (uint8_t k = 0; k < depth; k++) w *= 0.6f;
        for (uint32_t d = 0; d < gl->semantic_dim; d++) {
            out[d] += t->semantic_vector[d] * w;
        }
        total_weight += w;
    }
    if (total_weight > 1e-8f) {
        for (uint32_t d = 0; d < gl->semantic_dim; d++) out[d] /= total_weight;
    }
    normalize_vector(out, gl->semantic_dim);
}

int grounded_language_push_turn(grounded_language_t* gl,
                                 const float* semantic_vec,
                                 uint32_t vec_dim,
                                 uint32_t n_words,
                                 bool is_user) {
    if (!gl) return -1;
    if (gl->discourse.capacity == 0) gl->discourse.capacity = GL_DISCOURSE_MAX_TURNS;

    /* Pick the destination slot — append while not full, otherwise
     * overwrite the oldest (at head) and advance head one step. */
    uint8_t slot;
    if (gl->discourse.count < gl->discourse.capacity) {
        slot = (uint8_t)((gl->discourse.head + gl->discourse.count)
                         % gl->discourse.capacity);
        gl->discourse.count++;
    } else {
        slot = gl->discourse.head;
        gl->discourse.head = (uint8_t)((gl->discourse.head + 1)
                                        % gl->discourse.capacity);
        /* Free the prior occupant's vector before overwriting. */
        nimcp_free(gl->discourse.turns[slot].semantic_vector);
        gl->discourse.turns[slot].semantic_vector = NULL;
    }

    gl_discourse_turn_t* t = &gl->discourse.turns[slot];
    /* If the slot still has a stale vector from an out-of-order eviction
     * (defensive — paths above cleared it) free it first. */
    if (t->semantic_vector) {
        nimcp_free(t->semantic_vector);
        t->semantic_vector = NULL;
    }
    t->word_count   = n_words;
    t->timestamp_ms = (uint64_t)time(NULL) * 1000;
    t->is_user      = is_user;

    if (semantic_vec) {
        t->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
        if (!t->semantic_vector) {
            /* Roll back the slot reservation on alloc failure. */
            if (gl->discourse.count > 0) gl->discourse.count--;
            return -1;
        }
        uint32_t cmp = vec_dim < gl->semantic_dim ? vec_dim : gl->semantic_dim;
        memcpy(t->semantic_vector, semantic_vec, cmp * sizeof(float));
    }

    discourse_rebuild_context_blend(gl);
    return 0;
}

uint8_t grounded_language_get_discourse_turn_count(const grounded_language_t* gl) {
    if (!gl) return 0;
    return gl->discourse.count;
}

void grounded_language_set_discourse_capacity(grounded_language_t* gl,
                                                uint8_t capacity) {
    if (!gl) return;
    if (capacity < 1) capacity = 1;
    if (capacity > GL_DISCOURSE_MAX_TURNS) capacity = GL_DISCOURSE_MAX_TURNS;

    /* Capacity reduction: evict oldest turns (free their vectors) until
     * count fits. The newest `capacity` turns survive — copy them down
     * into [0, count) and reset head to 0 so the layout is canonical. */
    if (capacity < gl->discourse.capacity && gl->discourse.count > capacity) {
        uint8_t keep = capacity;
        gl_discourse_turn_t survivors[GL_DISCOURSE_MAX_TURNS];
        memset(survivors, 0, sizeof(survivors));

        /* Newest `keep` turns are at positions [count-keep, count). */
        uint8_t drop = (uint8_t)(gl->discourse.count - keep);
        for (uint8_t i = 0; i < drop; i++) {
            uint8_t idx = discourse_idx(&gl->discourse, i);
            nimcp_free(gl->discourse.turns[idx].semantic_vector);
            gl->discourse.turns[idx].semantic_vector = NULL;
        }
        for (uint8_t i = 0; i < keep; i++) {
            uint8_t src = discourse_idx(&gl->discourse, (uint8_t)(drop + i));
            survivors[i] = gl->discourse.turns[src];
            /* Defensive: if src and dst overlap and the source slot
             * gets memset below, we'd lose the pointer. Clear the
             * source by hand once we've copied. */
            gl->discourse.turns[src].semantic_vector = NULL;
        }
        /* Wipe the whole array (all heap pointers either freed or
         * moved to survivors[]) and reinstate the survivors at [0, keep). */
        memset(gl->discourse.turns, 0, sizeof(gl->discourse.turns));
        for (uint8_t i = 0; i < keep; i++) {
            gl->discourse.turns[i] = survivors[i];
        }
        gl->discourse.head  = 0;
        gl->discourse.count = keep;
    }

    gl->discourse.capacity = capacity;
    discourse_rebuild_context_blend(gl);
}

/*=============================================================================
 * Grounding (Cross-modal binding)
 *===========================================================================*/

int grounded_language_ground(grounded_language_t* gl, const gl_grounding_event_t* event) {
    if (!gl || !event || !event->word || !event->sensory_features) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_ground: NULL parameter");
        return -1;
    }

    /* #7 Negative grounding (anti-example).
     * "This is NOT a cat" — weaken the closest existing binding for the
     * matching modality without creating new entries / concepts. The
     * lookup is read-only: if the word isn't in the lexicon there's no
     * binding to weaken, so we return 0 (no-op success). The bus still
     * fires GROUNDED with negative confidence so curriculum modules can
     * count the contrast event. */
    if (event->negative) {
        char lower_w[GL_MAX_WORD_LEN];
        lowercase_word(event->word, lower_w, GL_MAX_WORD_LEN);
        const gl_lexicon_entry_t* found = lexicon_find(gl, lower_w);
        if (found) {
            gl_lexicon_entry_t* mut = (gl_lexicon_entry_t*)found;
            /* Pick binding with highest modality strength on this modality
             * (the one most "implicated" — the anti-example most
             * naturally targets the strongest competing association). */
            uint32_t target = UINT32_MAX;
            float best = 0.0f;
            for (uint32_t i = 0; i < mut->binding_count; i++) {
                float ms = mut->bindings[i].modality_strength[event->modality];
                if (ms > best) { best = ms; target = i; }
            }
            if (target == UINT32_MAX && mut->binding_count > 0) {
                /* No modality-specific signal — fall back to overall
                 * strongest binding. */
                target = 0;
                for (uint32_t i = 1; i < mut->binding_count; i++) {
                    if (mut->bindings[i].strength > mut->bindings[target].strength)
                        target = i;
                }
            }
            if (target != UINT32_MAX) {
                gl_word_binding_t* b = &mut->bindings[target];
                float decay = gl->hebbian_lr * event->attention;
                b->modality_strength[event->modality] -= decay;
                if (b->modality_strength[event->modality] < 0.0f)
                    b->modality_strength[event->modality] = 0.0f;
                b->strength -= decay * 0.5f;  /* overall weaker than mod-specific */
                if (b->strength < 0.0f) b->strength = 0.0f;
                b->last_activation_ms = (uint64_t)time(NULL) * 1000;
            }
        }

        gl->stats.total_negative_groundings++;

        /* Fire GROUNDED with negative confidence to mark the contrast. */
        extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
        gl_event_t bus_ev = {0};
        bus_ev.type        = GL_EVENT_GROUNDED;
        bus_ev.word        = event->word;
        bus_ev.concept_id  = 0;
        bus_ev.valence     = event->emotional_valence;
        bus_ev.arousal     = event->emotional_arousal;
        bus_ev.confidence  = -event->attention;   /* sign = anti-learning */
        bus_ev.semantic_vec = event->sensory_features;
        gl_fire_event(gl, &bus_ev);
        return 0;
    }

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

    /* Cortex modulation: when the in-modality cortex is currently active
     * (e.g. visual_cortex just processed an image while we're grounding
     * a visual word), boost binding strength up to +30%. Cheap scalar
     * read; no-op when cortexes aren't connected. */
    extern float gl_cortex_modulation_for_modality(grounded_language_t*,
                                                    gl_modality_t, float);
    bind_strength *= gl_cortex_modulation_for_modality(gl, event->modality, 0.3f);

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

    /* Memory-bridge fan-out: push the event into working memory,
     * episodic replay, and hippocampus if any of them are connected.
     * No-op when the brain init didn't wire those subsystems. */
    extern void gl_dispatch_event_to_memory(grounded_language_t*,
                                             const gl_grounding_event_t*,
                                             uint64_t);
    gl_dispatch_event_to_memory(gl, event, concept_id);

    /* Fire GROUNDED event on the cognitive bus. */
    extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
    gl_event_t bus_ev = {0};
    bus_ev.type        = GL_EVENT_GROUNDED;
    bus_ev.word        = event->word;
    bus_ev.concept_id  = concept_id;
    bus_ev.valence     = event->emotional_valence;
    bus_ev.arousal     = event->emotional_arousal;
    bus_ev.confidence  = event->attention;
    bus_ev.semantic_vec = event->sensory_features;
    gl_fire_event(gl, &bus_ev);

    return rc;
}

int grounded_language_learn_from_text(grounded_language_t* gl, const char* text) {
    if (!gl || !text) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_learn_from_text: NULL parameter (gl=%p, text=%p)",
            (void*)gl, (void*)text);
        return -1;
    }

    size_t text_len = strlen(text);
    char* buf = (char*)nimcp_malloc(text_len + 1);
    if (!buf) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_learn_from_text: failed to allocate buffer");
        return -1;
    }
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

    /* #9 Compositional templates — track frequent bigrams + trigrams. */
    _gl_track_phrases(gl, words, word_count);

    nimcp_free(buf);
    return updates;
}

uint64_t grounded_language_fast_map(grounded_language_t* gl, const char* word,
                                     const float* concept_features, uint32_t feature_dim,
                                     uint32_t category) {
    if (!gl || !word || !concept_features) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_fast_map: NULL parameter");
        return 0;
    }

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
    (void)mode;
    if (!gl || !intent || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_produce: NULL parameter (gl=%p, intent=%p, result=%p)",
            (void*)gl, (void*)intent, (void*)result);
        return -1;
    }
    memset(result, 0, sizeof(*result));

    /* SNN bridge is the only production path. The brain learns syntactic
     * structure emergently through training; we no longer impose templates.
     * If the bridge can't produce (untrained, or no concepts activated),
     * we return -1 and the caller's IDK gate fires "I don't know" — which
     * is the honest answer for an undertrained model. */
    if (!gl->snn_bridge) return -1;

    snn_lang_production_result_t spike_result;
    memset(&spike_result, 0, sizeof(spike_result));
    int rc = snn_language_bridge_produce(gl->snn_bridge, intent, intent_dim,
                                         &spike_result);
    if (rc != 0 || !spike_result.text || spike_result.word_count == 0) {
        snn_lang_production_result_cleanup(&spike_result);
        return -1;
    }

    /* Transfer the spike-produced text into the result. */
    result->text = spike_result.text;
    spike_result.text = NULL;  /* Ownership transferred. */
    result->word_count = spike_result.word_count;
    result->fluency = spike_result.fluency;
    result->relevance = spike_result.spike_confidence;
    result->creativity = spike_result.creativity;

    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (result->semantic_vector) {
        uint32_t copy_dim = (intent_dim < gl->semantic_dim)
            ? intent_dim : gl->semantic_dim;
        memcpy(result->semantic_vector, intent, copy_dim * sizeof(float));
    }

    snn_lang_production_result_cleanup(&spike_result);
    gl->stats.total_productions++;

    /* Fire PRODUCED event on the cognitive bus. */
    extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
    gl_event_t bus_ev = {0};
    bus_ev.type         = GL_EVENT_PRODUCED;
    bus_ev.text         = result->text;
    bus_ev.semantic_vec = result->semantic_vector;
    bus_ev.confidence   = result->fluency;
    gl_fire_event(gl, &bus_ev);

    return 0;
}

int grounded_language_describe_concept(grounded_language_t* gl, uint64_t concept_id,
                                        gl_production_result_t* result) {
    if (!gl || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_describe_concept: NULL parameter");
        return -1;
    }

    const float* features = get_concept_features(gl, concept_id);
    if (!features) return -1;

    return grounded_language_produce(gl, features, gl->semantic_dim,
                                      GL_PRODUCE_DESCRIBE, result);
}

int grounded_language_blend(grounded_language_t* gl,
                             uint64_t concept_a, uint64_t concept_b,
                             const float* vector_a, const float* vector_b,
                             uint32_t vec_dim, float blend_ratio,
                             gl_production_result_t* result) {
    if (!gl || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_blend: NULL parameter");
        return -1;
    }

    uint32_t dim = gl->semantic_dim;

    /* Get feature vectors */
    const float* fa = vector_a;
    const float* fb = vector_b;

    if (concept_a != 0) fa = get_concept_features(gl, concept_a);
    if (concept_b != 0) fb = get_concept_features(gl, concept_b);

    if (!fa && !fb) return -1;

    /* Blend vectors */
    float* blended = (float*)nimcp_calloc(dim, sizeof(float));
    if (!blended) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_blend: failed to allocate blend vector");
        return -1;
    }

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
    if (!gl || !result || num_sentences == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_narrate: NULL parameter or zero sentences");
        return -1;
    }
    memset(result, 0, sizeof(*result));

    if (num_sentences > 10) num_sentences = 10; /* Cap for sanity */

    size_t buf_size = num_sentences * 256;
    char* buf = (char*)nimcp_calloc(buf_size, 1);
    if (!buf) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_narrate: failed to allocate narrative buffer");
        return -1;
    }

    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!result->semantic_vector) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_narrate: failed to allocate semantic vector");
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
    if (!gl || !input_text || !response || response_max == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_respond: NULL parameter or zero response_max");
        return -1;
    }

    /* Comprehend input — gives us the semantic vector to drive production. */
    gl_comprehension_result_t comp = {0};
    int comp_rc = grounded_language_comprehend(gl, input_text, &comp);

    /* Produce via the SNN bridge. No template fallbacks, no quality-gated
     * stock phrases ("I understand about X.", "I don't have words for that
     * yet.", "I am learning."). Bridge succeeds → emit its output; bridge
     * fails → emit "I don't know.", the curriculum-trained uncertainty
     * marker (DK-B). The brain learns grammar emergently or admits it
     * can't — there is no third option. */
    gl_production_result_t prod = {0};
    int rc = -1;
    if (comp_rc == 0 && comp.semantic_vector) {
        rc = grounded_language_produce(gl, comp.semantic_vector, gl->semantic_dim,
                                        GL_PRODUCE_RESPOND, &prod);
    }

    if (rc == 0 && prod.text && prod.word_count > 0) {
        strncpy(response, prod.text, response_max - 1);
        response[response_max - 1] = '\0';
        if (confidence) *confidence = prod.fluency * prod.relevance;
    } else {
        snprintf(response, response_max, "I don't know.");
        if (confidence) *confidence = 0.0f;
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
    if (!gl || !input_text || !target_text) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_learn_pair: NULL parameter");
        return -1.0f;
    }

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
    /* Templates removed. The brain learns syntactic structure emergently via
     * SNN bridge plasticity (spike→word bindings + context vectors). No-op
     * kept so existing callers compile. */
    (void)gl;
    (void)text;
    return 0;
}

/*=============================================================================
 * Cross-modal Connections
 *===========================================================================*/

void grounded_language_connect_visual(grounded_language_t* gl, void* vis_ctx) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_visual: NULL gl");
        return;
    }
    gl->visual_ctx = vis_ctx;
}

void grounded_language_connect_auditory(grounded_language_t* gl, void* aud_ctx) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_auditory: NULL gl");
        return;
    }
    gl->auditory_ctx = aud_ctx;
}

void grounded_language_connect_speech(grounded_language_t* gl, void* speech_ctx) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_speech: NULL gl");
        return;
    }
    gl->speech_ctx = speech_ctx;
}

void grounded_language_connect_columns(grounded_language_t* gl, void* col_pool) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_columns: NULL gl");
        return;
    }
    gl->column_pool = col_pool;
}

void grounded_language_connect_emotional(grounded_language_t* gl, void* emo_ctx) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_emotional: NULL gl");
        return;
    }
    gl->emotional_ctx = emo_ctx;
}

/* Forward decl — implemented in nimcp_grounded_language_nlp.c. */
extern int gl_word_emb_for_bridge_external(void* ctx,
                                            const char* word_form,
                                            float* out_vec,
                                            uint32_t out_dim);

void grounded_language_connect_snn_bridge(grounded_language_t* gl,
                                           struct snn_language_bridge* bridge) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_snn_bridge: NULL gl");
        return;
    }
    gl->snn_bridge = bridge;

    /* PA-5: if embeddings are already wired on gl, push the lookup down
     * to the freshly-attached bridge so a later glove_blend > 0 can use
     * the cosine-emb signal without re-attaching. */
    if (bridge && gl->embeddings && gl->word_to_id_fn && gl->emb_dim > 0) {
        snn_language_bridge_set_embedding_lookup(bridge,
                                                   gl_word_emb_for_bridge_external,
                                                   gl, gl->emb_dim);
    }
}

/* Walkthrough round 3 fix: shared counter for the cold-start guard in both
 * flat and Riemannian learn_next_token_pair variants. Process-global
 * (single brain in process is the common case); guarded by atomic. */
static _Atomic uint64_t g_next_token_cold_start_skips = 0;

static void gl_next_token_cold_start_skips_inc(const char* prev, const char* next) {
    uint64_t v = (uint64_t)__atomic_add_fetch(&g_next_token_cold_start_skips, 1,
                                                __ATOMIC_RELAXED);
    /* Periodic visibility: every 10000th skip emit a single line so a
     * trainer running into "all my updates fail" sees it in the log. */
    if ((v % 10000ULL) == 0) {
        LOG_DEBUG(LOG_MODULE,
                  "learn_next_token_pair cold-start skips=%llu (latest pair: '%s' → '%s')",
                  (unsigned long long)v,
                  prev ? prev : "(null)", next ? next : "(null)");
    }
}

uint64_t grounded_language_next_token_cold_start_skips(void) {
    return (uint64_t)__atomic_load_n(&g_next_token_cold_start_skips, __ATOMIC_RELAXED);
}

uint64_t grounded_language_rebind_all_to_snn_bridge(grounded_language_t* gl) {
    if (!gl) return 0;
    if (!gl->snn_bridge) return 0;        /* no bridge attached → no-op */
    if (!gl->vocab_list || gl->vocab_count == 0) return 0;

    uint64_t mirrored = 0;
    for (uint32_t i = 0; i < gl->vocab_count; i++) {
        const gl_lexicon_entry_t* e = gl->vocab_list[i];
        if (!e || e->binding_count == 0) continue;
        for (uint32_t b = 0; b < e->binding_count; b++) {
            mirror_binding_to_bridge(gl, e,
                                      e->bindings[b].concept_id,
                                      e->bindings[b].strength);
            mirrored++;
        }
    }
    LOG_INFO(LOG_MODULE,
             "rebind_all_to_snn_bridge: mirrored %llu bindings across %u words",
             (unsigned long long)mirrored, gl->vocab_count);
    return mirrored;
}

/* ============================================================================
 * PA-4: Next-token contrastive training on the bridge binding matrix.
 *
 * For each (prev, next) bigram:
 *   1. Encode prev → concept_acts (the bridge's reverse lookup over its
 *      existing bindings to prev_word_pop).
 *   2. Decode top-K candidate words via the existing cosine score.
 *   3. If next_word is not the top-1, apply LTP to bindings (c, next_word)
 *      for active concepts c, and mild LTD to bindings (c, top_word) for
 *      the false winner — a one-step approximation of softmax cross-entropy
 *      gradient descent restricted to the target row and top-1 false
 *      winner row.
 *
 * No-op when bridge is unattached, prev has no encoding (cold-start),
 * lr is non-positive, or either word can't be hashed into the lexicon.
 * ============================================================================ */
int grounded_language_learn_next_token_pair(grounded_language_t* gl,
                                              const char* prev_word,
                                              const char* next_word,
                                              float lr)
{
    if (!gl || !prev_word || !next_word) return -1;
    if (!gl->snn_bridge) return -1;
    if (!isfinite(lr) || lr <= 0.0f) return -1;

    /* Find/create lexicon entries (assigns form_hash). */
    gl_lexicon_entry_t* prev_e =
        gl_internal_lexicon_find_or_create(gl, prev_word);
    gl_lexicon_entry_t* next_e =
        gl_internal_lexicon_find_or_create(gl, next_word);
    if (!prev_e || !next_e) return -1;

    uint32_t prev_word_pop = prev_e->form_hash % SNN_LANG_MAX_WORD_POPS;
    uint32_t next_word_pop = next_e->form_hash % SNN_LANG_MAX_WORD_POPS;

    /* Make sure both word_pops are registered with the bridge. */
    snn_language_bridge_register_word(gl->snn_bridge, prev_word_pop,
                                       prev_e->form);
    snn_language_bridge_register_word(gl->snn_bridge, next_word_pop,
                                       next_e->form);

    /* Encode prev_word → concept_acts. Heap allocation since 4096 floats
     * (16 KB) is too large for the stack. */
    const uint32_t n_concepts = SNN_LANG_MAX_CONCEPT_POPS;
    float* concept_acts = (float*)nimcp_calloc(n_concepts, sizeof(float));
    if (!concept_acts) return -1;

    if (snn_language_bridge_encode_word(gl->snn_bridge, prev_word_pop,
                                         concept_acts, n_concepts) != 0) {
        nimcp_free(concept_acts);
        return -1;
    }

    /* Cold-start guard: prev_word has no bindings yet. Nothing to condition
     * on. The first comprehension pass for prev_word will create bindings,
     * after which subsequent calls of this function become trainable.
     *
     * Walkthrough round 3 fix: count cold-start skips so the trainer can
     * see "all my pair updates failed silently" via a public stat instead
     * of guessing why the model isn't learning. Periodic LOG_DEBUG every
     * 10000 skips for in-the-loop visibility. */
    float total_act = 0.0f;
    for (uint32_t c = 0; c < n_concepts; c++) total_act += concept_acts[c];
    if (total_act <= 1e-6f) {
        gl_next_token_cold_start_skips_inc(prev_word, next_word);
        nimcp_free(concept_acts);
        return -1;
    }

    /* Decode top-K candidates given prev's encoding. */
    snn_lang_word_result_t topK[16];
    uint32_t num_out = 0;
    int rc = snn_language_bridge_decode_spikes(gl->snn_bridge, concept_acts,
                                                 n_concepts, topK,
                                                 16, &num_out);
    if (rc != 0) {
        nimcp_free(concept_acts);
        return -1;
    }

    /* Find rank of next_word_pop in the top-K (-1 if outside). */
    int target_rank = -1;
    for (uint32_t k = 0; k < num_out; k++) {
        if (topK[k].word_pop == next_word_pop) {
            target_rank = (int)k;
            break;
        }
    }

    /* LTP on (c, next_word_pop) — pull the target word toward the prefix's
     * concept activation profile. Active-only: skip near-zero concepts so
     * we don't waste a binding on noise. The activation threshold mirrors
     * the existing 0.05 heuristic used elsewhere in the bridge. */
    const float ltp_threshold = 0.05f;
    for (uint32_t c = 0; c < n_concepts; c++) {
        if (concept_acts[c] > ltp_threshold) {
            float delta = lr * concept_acts[c];
            snn_language_bridge_strengthen_binding(gl->snn_bridge,
                                                    c, next_word_pop, delta);
        }
    }

    /* LTD on (c, top-1 false winner) when target wasn't already #1. Half
     * the LR so LTD is gentler than LTP — biology and softmax CE both
     * prefer asymmetric updates in this direction. Skip if top-1 is the
     * target itself (already learned correctly). */
    if (target_rank != 0 && num_out > 0) {
        uint32_t false_winner = topK[0].word_pop;
        if (false_winner != next_word_pop) {
            float ltd_lr = -0.5f * lr;
            for (uint32_t c = 0; c < n_concepts; c++) {
                if (concept_acts[c] > ltp_threshold) {
                    snn_language_bridge_strengthen_binding(gl->snn_bridge,
                                                            c, false_winner,
                                                            ltd_lr * concept_acts[c]);
                }
            }
        }
    }

    nimcp_free(concept_acts);
    return 0;
}

/* ============================================================================
 * PA-4+: Riemannian / sigmoid-reparameterized next-token training.
 *
 * Same overall contrastive structure as grounded_language_learn_next_token_pair:
 *   1. Encode prev → concept_acts.
 *   2. Decode top-K and find target rank.
 *   3. LTP on (c, next_word_pop), LTD on (c, top-1 false winner).
 *
 * The only difference is each binding update goes through
 * snn_language_bridge_strengthen_binding_riemannian — Fisher-preconditioned
 * step in u-space (w = σ(u)), so binding weights near 0 or 1 are damped
 * automatically instead of being clipped post-hoc.
 *
 * Default OFF: callers must invoke this *_riemannian variant explicitly.
 * The flat learn_next_token_pair path remains bit-for-bit unchanged.
 * ============================================================================ */
int grounded_language_learn_next_token_pair_riemannian(grounded_language_t* gl,
                                                        const char* prev_word,
                                                        const char* next_word,
                                                        float lr)
{
    if (!gl || !prev_word || !next_word) return -1;
    if (!gl->snn_bridge) return -1;
    if (!isfinite(lr) || lr <= 0.0f) return -1;

    gl_lexicon_entry_t* prev_e =
        gl_internal_lexicon_find_or_create(gl, prev_word);
    gl_lexicon_entry_t* next_e =
        gl_internal_lexicon_find_or_create(gl, next_word);
    if (!prev_e || !next_e) return -1;

    uint32_t prev_word_pop = prev_e->form_hash % SNN_LANG_MAX_WORD_POPS;
    uint32_t next_word_pop = next_e->form_hash % SNN_LANG_MAX_WORD_POPS;

    snn_language_bridge_register_word(gl->snn_bridge, prev_word_pop,
                                       prev_e->form);
    snn_language_bridge_register_word(gl->snn_bridge, next_word_pop,
                                       next_e->form);

    const uint32_t n_concepts = SNN_LANG_MAX_CONCEPT_POPS;
    float* concept_acts = (float*)nimcp_calloc(n_concepts, sizeof(float));
    if (!concept_acts) return -1;

    if (snn_language_bridge_encode_word(gl->snn_bridge, prev_word_pop,
                                         concept_acts, n_concepts) != 0) {
        nimcp_free(concept_acts);
        return -1;
    }

    float total_act = 0.0f;
    for (uint32_t c = 0; c < n_concepts; c++) total_act += concept_acts[c];
    if (total_act <= 1e-6f) {
        nimcp_free(concept_acts);
        return -1;
    }

    snn_lang_word_result_t topK[16];
    uint32_t num_out = 0;
    int rc = snn_language_bridge_decode_spikes(gl->snn_bridge, concept_acts,
                                                 n_concepts, topK,
                                                 16, &num_out);
    if (rc != 0) {
        nimcp_free(concept_acts);
        return -1;
    }

    int target_rank = -1;
    for (uint32_t k = 0; k < num_out; k++) {
        if (topK[k].word_pop == next_word_pop) {
            target_rank = (int)k;
            break;
        }
    }

    /* LTP on (c, next_word_pop) — Riemannian step. `grad` here is
     * lr * concept_acts[c], same as the flat path's delta; the bridge
     * function applies the Fisher preconditioner internally. */
    const float ltp_threshold = 0.05f;
    for (uint32_t c = 0; c < n_concepts; c++) {
        if (concept_acts[c] > ltp_threshold) {
            float grad = lr * concept_acts[c];
            snn_language_bridge_strengthen_binding_riemannian(gl->snn_bridge,
                                                               c, next_word_pop,
                                                               grad);
        }
    }

    if (target_rank != 0 && num_out > 0) {
        uint32_t false_winner = topK[0].word_pop;
        if (false_winner != next_word_pop) {
            float ltd_lr = -0.5f * lr;
            for (uint32_t c = 0; c < n_concepts; c++) {
                if (concept_acts[c] > ltp_threshold) {
                    snn_language_bridge_strengthen_binding_riemannian(
                        gl->snn_bridge, c, false_winner,
                        ltd_lr * concept_acts[c]);
                }
            }
        }
    }

    nimcp_free(concept_acts);
    return 0;
}

/*=============================================================================
 * PA-4+ : optional bigram-spectrum tracker
 *
 * The spectrum lives in caller-owned memory and is referenced from a
 * small static side-map keyed by gl pointer. We avoid touching the
 * grounded_language struct layout (the persistence sidecar walks it
 * field-by-field via nimcp_grounded_language_internal.h), so an opt-in
 * side-channel keeps the layout untouched while still letting the
 * trainer attach a diagnostic.
 *
 * Cap is intentionally tiny — at most one spectrum per active brain,
 * and brains rarely exceed two in the same process.
 *===========================================================================*/

#define GL_SPECTRUM_MAP_CAP 4

static struct {
    grounded_language_t* gl;
    bigram_spectrum_t*   spectrum;
} g_spectrum_map[GL_SPECTRUM_MAP_CAP] = { {0} };

/* Walkthrough round 3 fix: serialize all access to g_spectrum_map. The map
 * is accessed from training threads (learn_text_bigrams) and concurrent
 * attach/detach from RPC handlers. Without this lock, two threads racing
 * on insert can leave the map in an inconsistent state, and a learn-loop
 * reader can observe a dangling slot mid-update. */
static pthread_mutex_t g_spectrum_map_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_spectrum_map_overflow_warns = 0;

static bigram_spectrum_t* gl_get_attached_spectrum(grounded_language_t* gl) {
    if (!gl) return NULL;
    pthread_mutex_lock(&g_spectrum_map_lock);
    bigram_spectrum_t* found = NULL;
    for (int i = 0; i < GL_SPECTRUM_MAP_CAP; i++) {
        if (g_spectrum_map[i].gl == gl) {
            found = g_spectrum_map[i].spectrum;
            break;
        }
    }
    pthread_mutex_unlock(&g_spectrum_map_lock);
    return found;
}

/* Walkthrough round 3 fix: called from grounded_language_destroy so the
 * map can never hand out a spectrum bound to a freed gl pointer (UAF
 * if the OS later reuses the same address for a fresh gl). */
static void gl_detach_spectrum_for_destroy(grounded_language_t* gl) {
    if (!gl) return;
    pthread_mutex_lock(&g_spectrum_map_lock);
    for (int i = 0; i < GL_SPECTRUM_MAP_CAP; i++) {
        if (g_spectrum_map[i].gl == gl) {
            g_spectrum_map[i].gl = NULL;
            g_spectrum_map[i].spectrum = NULL;
        }
    }
    pthread_mutex_unlock(&g_spectrum_map_lock);
}

void grounded_language_attach_bigram_spectrum(grounded_language_t* gl,
                                                void* spectrum) {
    if (!gl) return;
    bigram_spectrum_t* bs = (bigram_spectrum_t*)spectrum;

    pthread_mutex_lock(&g_spectrum_map_lock);

    /* Replace existing entry if any. */
    for (int i = 0; i < GL_SPECTRUM_MAP_CAP; i++) {
        if (g_spectrum_map[i].gl == gl) {
            g_spectrum_map[i].spectrum = bs;
            if (!bs) g_spectrum_map[i].gl = NULL;  /* detach */
            pthread_mutex_unlock(&g_spectrum_map_lock);
            return;
        }
    }
    if (!bs) {
        pthread_mutex_unlock(&g_spectrum_map_lock);
        return;  /* nothing to detach */
    }

    /* Insert into first free slot. */
    for (int i = 0; i < GL_SPECTRUM_MAP_CAP; i++) {
        if (g_spectrum_map[i].gl == NULL) {
            g_spectrum_map[i].gl = gl;
            g_spectrum_map[i].spectrum = bs;
            pthread_mutex_unlock(&g_spectrum_map_lock);
            return;
        }
    }
    /* Walkthrough round 3 fix: capacity overflow is now logged + counted
     * (was silently dropped). Operators can see we ran out of slots. */
    g_spectrum_map_overflow_warns++;
    pthread_mutex_unlock(&g_spectrum_map_lock);
    LOG_WARN(LOG_MODULE,
             "attach_bigram_spectrum: map full (cap=%d); dropped attach for gl=%p",
             GL_SPECTRUM_MAP_CAP, (void*)gl);
}

uint64_t grounded_language_bigram_spectrum_map_overflow_warns(void) {
    pthread_mutex_lock(&g_spectrum_map_lock);
    uint64_t v = g_spectrum_map_overflow_warns;
    pthread_mutex_unlock(&g_spectrum_map_lock);
    return v;
}

/*=============================================================================
 * Tier-1 #2: Rule-based anaphora / pronoun resolution (v1).
 *
 * Side-map keyed by gl pointer (same pattern as g_spectrum_map) so we
 * don't perturb the grounded_language struct layout — the persistence
 * sidecar walks that struct field-by-field via the internal header and
 * any layout drift breaks checkpoint compatibility.
 *
 * Each entry holds an 8-deep referent ring + the toggle. The pronoun
 * table itself is process-global static const data. Successful resolves
 * bump g_anaphora_resolutions atomically.
 *===========================================================================*/

#define GL_ANAPHORA_MAP_CAP        4
#define GL_ANAPHORA_RING_CAP       8

typedef enum {
    GL_GENDER_UNKNOWN = 0,
    GL_GENDER_SINGULAR_MALE,
    GL_GENDER_SINGULAR_FEMALE,
    GL_GENDER_SINGULAR_NEUTER,
    GL_GENDER_PLURAL,
} gl_anaphora_gender_t;

typedef struct {
    char                   form[GL_MAX_WORD_LEN];   /* lowercased */
    gl_anaphora_gender_t   gender;
    uint64_t               timestamp_us;
} gl_anaphora_referent_t;

typedef struct {
    bool                   enabled;
    gl_anaphora_referent_t ring[GL_ANAPHORA_RING_CAP];
    uint32_t               head;   /* next slot to write — push location */
    uint32_t               size;   /* number of valid entries (≤ cap) */
} gl_anaphora_state_t;

/* Pronoun table — small enough that linear scan beats any hash. */
typedef struct {
    const char*          word;
    gl_anaphora_gender_t gender;
} gl_anaphora_pronoun_entry_t;

static const gl_anaphora_pronoun_entry_t g_pronouns[] = {
    { "he",     GL_GENDER_SINGULAR_MALE   },
    { "him",    GL_GENDER_SINGULAR_MALE   },
    { "his",    GL_GENDER_SINGULAR_MALE   },
    { "she",    GL_GENDER_SINGULAR_FEMALE },
    { "her",    GL_GENDER_SINGULAR_FEMALE },
    { "hers",   GL_GENDER_SINGULAR_FEMALE },
    { "it",     GL_GENDER_SINGULAR_NEUTER },
    { "this",   GL_GENDER_SINGULAR_NEUTER },
    { "that",   GL_GENDER_SINGULAR_NEUTER },
    { "they",   GL_GENDER_PLURAL          },
    { "them",   GL_GENDER_PLURAL          },
    { "their",  GL_GENDER_PLURAL          },
    { "theirs", GL_GENDER_PLURAL          },
    { "these",  GL_GENDER_PLURAL          },
    { "those",  GL_GENDER_PLURAL          },
};
#define GL_PRONOUN_COUNT (sizeof(g_pronouns) / sizeof(g_pronouns[0]))

static struct {
    grounded_language_t* gl;
    gl_anaphora_state_t* state;
} g_anaphora_map[GL_ANAPHORA_MAP_CAP] = { {0} };

static pthread_mutex_t g_anaphora_map_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t        g_anaphora_resolutions = 0;  /* lock-protected */

/* Map operations. The map lock protects both the slot table AND the
 * per-gl state's ring (we only ever touch state pointers under the
 * lock; readers of the ring also hold the lock). */

static gl_anaphora_state_t* anaphora_lookup_locked(grounded_language_t* gl) {
    for (int i = 0; i < GL_ANAPHORA_MAP_CAP; i++) {
        if (g_anaphora_map[i].gl == gl) return g_anaphora_map[i].state;
    }
    return NULL;
}

static gl_anaphora_state_t* anaphora_get_or_create_locked(grounded_language_t* gl) {
    gl_anaphora_state_t* st = anaphora_lookup_locked(gl);
    if (st) return st;
    /* Insert into first free slot. */
    for (int i = 0; i < GL_ANAPHORA_MAP_CAP; i++) {
        if (g_anaphora_map[i].gl == NULL) {
            st = (gl_anaphora_state_t*)nimcp_calloc(1, sizeof(*st));
            if (!st) return NULL;
            g_anaphora_map[i].gl = gl;
            g_anaphora_map[i].state = st;
            return st;
        }
    }
    /* Map full — caller will see NULL and bail. */
    return NULL;
}

static void gl_detach_anaphora_for_destroy(grounded_language_t* gl) {
    if (!gl) return;
    pthread_mutex_lock(&g_anaphora_map_lock);
    for (int i = 0; i < GL_ANAPHORA_MAP_CAP; i++) {
        if (g_anaphora_map[i].gl == gl) {
            nimcp_free(g_anaphora_map[i].state);
            g_anaphora_map[i].gl = NULL;
            g_anaphora_map[i].state = NULL;
        }
    }
    pthread_mutex_unlock(&g_anaphora_map_lock);
}

bool grounded_language_set_anaphora_enabled(grounded_language_t* gl,
                                              bool enabled) {
    if (!gl) return false;
    pthread_mutex_lock(&g_anaphora_map_lock);
    gl_anaphora_state_t* st = anaphora_get_or_create_locked(gl);
    if (!st) {
        pthread_mutex_unlock(&g_anaphora_map_lock);
        return false;
    }
    st->enabled = enabled;
    /* Disabling clears the ring so a later re-enable doesn't carry state
     * across an explicit OFF window. */
    if (!enabled) {
        st->head = 0;
        st->size = 0;
        memset(st->ring, 0, sizeof(st->ring));
    }
    pthread_mutex_unlock(&g_anaphora_map_lock);
    return true;
}

uint64_t grounded_language_anaphora_resolutions(void) {
    pthread_mutex_lock(&g_anaphora_map_lock);
    uint64_t v = g_anaphora_resolutions;
    pthread_mutex_unlock(&g_anaphora_map_lock);
    return v;
}

/* Pronoun lookup — returns gender, or GL_GENDER_UNKNOWN if word is not a
 * pronoun. Caller must already have a lowercased form. */
static gl_anaphora_gender_t anaphora_pronoun_gender(const char* lower_word) {
    if (!lower_word || !lower_word[0]) return GL_GENDER_UNKNOWN;
    for (size_t i = 0; i < GL_PRONOUN_COUNT; i++) {
        if (strcmp(g_pronouns[i].word, lower_word) == 0) {
            return g_pronouns[i].gender;
        }
    }
    return GL_GENDER_UNKNOWN;
}

/* Gender heuristic for content nouns. Pure recency-based; no syntactic
 * analysis.
 *   - The caller (`comprehend`) lowercases tokens before invoking this
 *     pass, so capitalization-based proper-noun detection is impossible
 *     here — that signal was lost upstream. Earlier versions had an
 *     `isupper(first)` branch that returned SINGULAR_MALE; it never
 *     fired and has been removed (INT-1 walkthrough finding).
 *   - Trailing 's' (length ≥ 4 to skip "is"/"as", and != "ss" to skip
 *     boss/glass) → plural.
 *   - Otherwise → singular_neuter.
 * SINGULAR_FEMALE / SINGULAR_MALE are signalled by gendered pronoun
 * cues ("she/her/hers" or "he/him/his") propagated forward by the
 * caller via `pending_female` / `pending_male`. So "He is Tom" and
 * "She is Alice" both work; bare "Tom" / "Alice" mid-sentence with no
 * cue stay neuter (v1 caveat — fixable later with a name lexicon).
 */
static gl_anaphora_gender_t anaphora_classify_noun(const char* surface_form) {
    if (!surface_form || !surface_form[0]) return GL_GENDER_SINGULAR_NEUTER;
    size_t len = strlen(surface_form);

    /* Plural detection by trailing 's' (length ≥ 4 avoids 1-letter and
     * common short stop-words like "is"/"as"). Also avoid "ss" doubling
     * (boss/glass → not plural). */
    if (len >= 4 && surface_form[len-1] == 's' && surface_form[len-2] != 's') {
        return GL_GENDER_PLURAL;
    }

    return GL_GENDER_SINGULAR_NEUTER;
}

/* Push a referent onto the ring (caller already holds the map lock). */
static void anaphora_push_referent(gl_anaphora_state_t* st,
                                     const char* lower_form,
                                     gl_anaphora_gender_t gender) {
    if (!st || !lower_form || !lower_form[0]) return;
    if (gender == GL_GENDER_UNKNOWN) gender = GL_GENDER_SINGULAR_NEUTER;

    gl_anaphora_referent_t* slot = &st->ring[st->head];
    memset(slot, 0, sizeof(*slot));
    size_t cap = sizeof(slot->form) - 1;
    strncpy(slot->form, lower_form, cap);
    slot->form[cap] = '\0';
    slot->gender = gender;
    /* Microsecond-ish timestamp — we only need monotonicity for tie-break
     * at backwards scan, and the ring index already encodes that, so
     * this is essentially decorative. */
    slot->timestamp_us = (uint64_t)time(NULL) * 1000000ull;

    st->head = (st->head + 1) % GL_ANAPHORA_RING_CAP;
    if (st->size < GL_ANAPHORA_RING_CAP) st->size++;
}

/* Resolve a pronoun by walking the ring backwards from most-recent.
 * Returns the form of the matched referent (pointing into the ring,
 * valid only while the lock is held — caller copies it out). NULL if
 * no gender match is found. */
static const char* anaphora_resolve_locked(gl_anaphora_state_t* st,
                                             gl_anaphora_gender_t pronoun_gender) {
    if (!st || st->size == 0) return NULL;
    /* head points to NEXT write slot. Most-recent is head-1 (mod cap). */
    for (uint32_t step = 0; step < st->size; step++) {
        uint32_t idx = (st->head + GL_ANAPHORA_RING_CAP - 1 - step)
                        % GL_ANAPHORA_RING_CAP;
        if (st->ring[idx].gender == pronoun_gender) {
            return st->ring[idx].form;
        }
    }
    return NULL;
}

/* The hook called from comprehend. Walks the token list, classifies each
 * token as pronoun or content noun, pushes referents, and (when a
 * pronoun resolves) folds the referent's lexicon-entry bindings into the
 * caller's activated_concepts/activation_levels arrays at half-strength.
 *
 * Returns the number of successful resolutions performed in this call.
 * No-op (and returns 0) when the resolver is disabled or unallocated.
 *
 * NOTE: caller passes the in/out concept_count by pointer. If we resolve
 * a pronoun but the activated_concepts array is full, we silently skip
 * the binding-fold for that pronoun (counter is still bumped — the
 * resolution itself succeeded; only the side-effect of activating
 * concepts was capped).
 */
static uint32_t anaphora_run_pass(grounded_language_t* gl,
                                    char** words,
                                    uint32_t word_count,
                                    uint64_t* activated_concepts,
                                    float* activation_levels,
                                    uint32_t* concept_count_io) {
    if (!gl || !words || word_count == 0) return 0;

    /* Lock-and-check enabled; if disabled, fast-path return without
     * touching the ring. */
    pthread_mutex_lock(&g_anaphora_map_lock);
    gl_anaphora_state_t* st = anaphora_lookup_locked(gl);
    if (!st || !st->enabled) {
        pthread_mutex_unlock(&g_anaphora_map_lock);
        return 0;
    }

    uint32_t resolved_here = 0;
    bool pending_female = false;  /* "she/her/hers" cue → next content noun is female */
    bool pending_male   = false;  /* "he/him/his"  cue → next content noun is male
                                   * (parallel to female cue — INT-1 fix; replaces
                                   *  dead capitalization-based MALE classification) */

    for (uint32_t w = 0; w < word_count; w++) {
        const char* surface = words[w];
        if (!surface || !surface[0]) continue;

        char lower[GL_MAX_WORD_LEN];
        lowercase_word(surface, lower, GL_MAX_WORD_LEN);

        gl_anaphora_gender_t pron = anaphora_pronoun_gender(lower);
        if (pron != GL_GENDER_UNKNOWN) {
            /* Pronoun. Try to resolve. */
            const char* matched_form = anaphora_resolve_locked(st, pron);
            if (matched_form) {
                /* Copy form before unlocking the map — though we hold the
                 * lock through the whole pass, so this is just defensive. */
                char form_copy[GL_MAX_WORD_LEN];
                size_t cap = sizeof(form_copy) - 1;
                strncpy(form_copy, matched_form, cap);
                form_copy[cap] = '\0';

                resolved_here++;
                g_anaphora_resolutions++;

                /* Fold the referent's lexicon-entry bindings (at half
                 * strength) into the caller's activated_concepts list.
                 * lexicon_find takes a const gl* and is read-only — safe
                 * to call while holding our lock (no overlap with the
                 * gl's own internal state). */
                if (activated_concepts && activation_levels && concept_count_io) {
                    const gl_lexicon_entry_t* ref_entry = lexicon_find(gl, form_copy);
                    if (ref_entry) {
                        for (uint32_t b = 0;
                             b < ref_entry->binding_count &&
                             *concept_count_io < GL_MAX_ACTIVE_CONCEPTS;
                             b++) {
                            const gl_word_binding_t* bd = &ref_entry->bindings[b];
                            if (bd->strength < GL_ASSOC_PRUNE_THRESHOLD) continue;

                            float weight = 0.5f * bd->strength;

                            /* Merge with existing activation if present. */
                            bool merged = false;
                            for (uint32_t c = 0; c < *concept_count_io; c++) {
                                if (activated_concepts[c] == bd->concept_id) {
                                    activation_levels[c] += weight;
                                    merged = true;
                                    break;
                                }
                            }
                            if (!merged) {
                                activated_concepts[*concept_count_io]  = bd->concept_id;
                                activation_levels[*concept_count_io]   = weight;
                                (*concept_count_io)++;
                            }
                        }
                    }
                }
            }
            /* Cue-update: a literal "she" / "her" / "hers" suggests the
             * next content noun (a NAME) refers to a female entity. We
             * propagate pending_female forward so something like
             * "She is Alice" makes Alice female on push. The symmetric
             * pending_male path covers "He is Tom" — added in the INT-1
             * fix when we removed the dead capitalization branch. */
            if (pron == GL_GENDER_SINGULAR_FEMALE) pending_female = true;
            if (pron == GL_GENDER_SINGULAR_MALE)   pending_male   = true;
            continue;
        }

        /* Not a pronoun — candidate referent.
         * Skip lexicon function words (the/a/an/of/and/...) so they
         * don't displace real nouns from the ring. The lexicon's
         * learned_class is a soft hint; if the entry doesn't exist
         * we still push the surface form (any unknown noun is more
         * useful than nothing for a recency-based v1).
         */
        const gl_lexicon_entry_t* ent = lexicon_find(gl, lower);
        if (ent && ent->learned_class == GL_CLASS_FUNCTION) continue;

        gl_anaphora_gender_t g = anaphora_classify_noun(surface);
        if (pending_female) {
            g = GL_GENDER_SINGULAR_FEMALE;
            pending_female = false;
            pending_male   = false;  /* female cue wins if both somehow set */
        } else if (pending_male) {
            g = GL_GENDER_SINGULAR_MALE;
            pending_male = false;
        }
        anaphora_push_referent(st, lower, g);
    }

    pthread_mutex_unlock(&g_anaphora_map_lock);
    return resolved_here;
}

int grounded_language_learn_text_bigrams(grounded_language_t* gl,
                                           const char* text, float lr)
{
    if (!gl || !text) return -1;
    if (!gl->snn_bridge) return 0;  /* no bridge → no-op (count 0) */
    if (!isfinite(lr) || lr <= 0.0f) return -1;

    size_t L = strlen(text);
    char* buf = (char*)nimcp_malloc(L + 1);
    if (!buf) return -1;
    memcpy(buf, text, L + 1);

    char* words[GL_MAX_PRODUCTION_WORDS];
    uint32_t n = tokenize_text(buf, words, GL_MAX_PRODUCTION_WORDS);

    bigram_spectrum_t* spectrum = gl_get_attached_spectrum(gl);
    uint32_t spec_cap = spectrum ? bigram_spectrum_vocab_cap(spectrum) : 0u;

    int processed = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        if (grounded_language_learn_next_token_pair(gl, words[i],
                                                      words[i + 1], lr) == 0) {
            processed++;

            /* PA-4+ : record the bigram into the attached spectrum.
             * Use the lexicon form_hash modded down to spectrum's vocab
             * cap. Out-of-range ids are silently ignored by record(). */
            if (spectrum && spec_cap > 0) {
                gl_lexicon_entry_t* pe =
                    gl_internal_lexicon_find_or_create(gl, words[i]);
                gl_lexicon_entry_t* ne =
                    gl_internal_lexicon_find_or_create(gl, words[i + 1]);
                if (pe && ne) {
                    uint32_t pid = pe->form_hash % spec_cap;
                    uint32_t nid = ne->form_hash % spec_cap;
                    bigram_spectrum_record(spectrum, pid, nid);
                }
            }
        }
    }

    nimcp_free(buf);
    return processed;
}

/*=============================================================================
 * #14 Dialect / accent conditioning
 *===========================================================================*/

void grounded_language_set_dialect(grounded_language_t* gl, const char* dialect) {
    if (!gl) return;
    if (!dialect || !dialect[0]) {
        gl->context_dialect[0] = '\0';
        return;
    }
    /* Truncating copy + guaranteed NUL. strncpy doesn't NUL-terminate
     * when src ≥ dst; do it explicitly. */
    size_t n = strlen(dialect);
    if (n >= GL_MAX_DIALECT_LEN) n = GL_MAX_DIALECT_LEN - 1;
    memcpy(gl->context_dialect, dialect, n);
    gl->context_dialect[n] = '\0';
}

const char* grounded_language_get_dialect(const grounded_language_t* gl) {
    if (!gl) return "";
    return gl->context_dialect;
}

/*=============================================================================
 * Query / Introspection
 *===========================================================================*/

const gl_lexicon_entry_t* grounded_language_lookup(const grounded_language_t* gl,
                                                     const char* word) {
    if (!gl || !word) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_lookup: NULL parameter");
        return NULL;
    }
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

    /* Forgetting telemetry (#15) — sum the 24-hour ring + cumulative. */
    uint32_t sum24 = 0;
    for (int i = 0; i < 24; i++) sum24 += gl->decayed_ring[i];
    stats->entries_decayed_last_24h = sum24;
    stats->entries_decayed_all_time = gl->decayed_all_time;
}

uint32_t grounded_language_get_semantic_dim(const grounded_language_t* gl) {
    return gl ? gl->semantic_dim : 0u;
}

int grounded_language_get_probe_metrics(const grounded_language_t* gl,
                                          gl_probe_metrics_t* out) {
    if (!gl || !out) return -1;
    memset(out, 0, sizeof(*out));

    /* Lexicon gauges — share the get_stats walk for binding averages. */
    float total_strength = 0.0f;
    float total_confidence = 0.0f;
    uint32_t total_bindings = 0;
    for (uint32_t w = 0; w < gl->vocab_count; w++) {
        const gl_lexicon_entry_t* entry = gl->vocab_list[w];
        if (!entry) continue;
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            total_strength   += entry->bindings[b].strength;
            total_confidence += entry->bindings[b].confidence;
            total_bindings++;
        }
    }
    out->vocab_count          = gl->vocab_count;
    out->templates_count      = 0;  /* templates removed — emergent grammar via SNN bridge */
    out->avg_binding_strength = (total_bindings > 0)
        ? total_strength / (float)total_bindings : 0.0f;
    out->avg_binding_confidence = (total_bindings > 0)
        ? total_confidence / (float)total_bindings : 0.0f;

    /* Bus state. */
    out->subscriber_count = gl->subscriber_count;
    for (uint32_t i = 0; i < gl->subscriber_count; i++) {
        if (gl->subscriber_priorities[i] > 0)
            out->subscriber_high_priority++;
        if (gl->subscriber_masks[i] != GL_EVENT_MASK_ALL)
            out->subscriber_filtered++;
    }
    out->events_dropped_reentry = gl->events_dropped_reentry;
    out->in_fire_event          = gl->in_fire_event;

    /* Forgetting curve. */
    uint32_t sum24 = 0;
    for (int i = 0; i < 24; i++) sum24 += gl->decayed_ring[i];
    out->entries_decayed_last_24h = sum24;
    out->entries_decayed_all_time = gl->decayed_all_time;

    /* Network bridge magnitudes (last broadcast). */
    out->last_lnn_response_mag = gl->last_lnn_mag;
    out->last_cnn_response_mag = gl->last_cnn_mag;
    out->last_fno_response_mag = gl->last_fno_mag;
    out->last_ann_response_mag = gl->last_ann_mag;

    /* Throughput counters from gl_stats_t. */
    out->total_groundings      = gl->stats.total_groundings;
    out->total_comprehensions  = gl->stats.total_comprehensions;
    out->total_productions     = gl->stats.total_productions;
    out->total_new_words       = gl->stats.vocab_size;

    /* Region wiring health (booleans for each connect_*). */
    out->broca_attached          = (gl->broca_adapter != NULL);
    out->wernicke_attached       = (gl->wernicke_adapter != NULL);
    out->working_memory_attached = (gl->working_memory != NULL);
    out->hippocampus_attached    = (gl->hippocampus != NULL);
    out->embedding_attached      = (gl->embeddings != NULL);
    out->tokenizer_attached      = (gl->tokenizer != NULL);

    /* Cortex modulation taps — read live from connected cortexes. */
    gl_cortex_modulation_t mod;
    if (grounded_language_get_cortex_modulation(
            (grounded_language_t*)gl, &mod) == 0) {
        out->visual_activity    = mod.visual_activity;
        out->audio_salience     = mod.audio_salience;
        out->speech_confidence  = mod.speech_confidence;
    }

    /* Dialect tag (#14) — copy with NUL guarantee. */
    size_t dlen = 0;
    while (dlen < GL_MAX_DIALECT_LEN - 1 && gl->context_dialect[dlen] != '\0') dlen++;
    memcpy(out->context_dialect, gl->context_dialect, dlen);
    out->context_dialect[dlen] = '\0';

    /* Negative-grounding + active-learning counters. */
    out->total_negative_groundings = gl->stats.total_negative_groundings;
    out->total_needs_grounding     = gl->stats.total_needs_grounding;
    return 0;
}

/*=============================================================================
 * Serialization (stub — full implementation deferred)
 *===========================================================================*/

int grounded_language_save(const grounded_language_t* gl, const char* path) {
    if (!gl || !path) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_save: NULL parameter");
        return -1;
    }

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    /* Magic + version. v2 adds context_dialect (#14). v3 adds
     * compositional phrases (#9). v4 (#6) compresses each entry's
     * context_vector to int8 with per-vector max-abs scaling — 4×
     * size reduction at <0.4% per-element error. v5 removes the
     * syntactic-template block; grammar is now learned emergently via
     * SNN bridge plasticity. v5 readers skip the legacy template
     * section when loading v4-and-earlier files. */
    uint32_t magic = GL_MAGIC;
    uint32_t version = 5;
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

        /* Context vector — v4 int8 quantization (#6): per-vector
         * max-abs as float, then int8 codes. Decoded as
         *   x = (q / 127.0) × max_abs.
         * Empty/zero vectors store max_abs=0 and skip the int8 block.
         * v3 and earlier wrote a float block — kept for read parity. */
        fwrite(&entry->context_initialized, sizeof(entry->context_initialized), 1, f);
        if (entry->context_initialized) {
            float max_abs = 0.0f;
            for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                float a = fabsf(entry->context_vector[d]);
                if (a > max_abs) max_abs = a;
            }
            fwrite(&max_abs, sizeof(float), 1, f);
            if (max_abs > 0.0f) {
                /* Quantize to int8. Heap-allocate so we never silently
                 * zero out user data when semantic_dim is bumped — the
                 * 4096-cap from load() bounds peak allocation here. */
                int8_t* qbuf = (int8_t*)nimcp_malloc(gl->semantic_dim);
                if (!qbuf) {
                    fclose(f);
                    return -1;
                }
                float inv = 127.0f / max_abs;
                for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                    float v = entry->context_vector[d] * inv;
                    if (v >  127.0f) v =  127.0f;
                    if (v < -127.0f) v = -127.0f;
                    qbuf[d] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
                }
                fwrite(qbuf, sizeof(int8_t), gl->semantic_dim, f);
                nimcp_free(qbuf);
            }
        }

        /* Bindings */
        fwrite(&entry->binding_count, sizeof(entry->binding_count), 1, f);
        for (uint32_t b = 0; b < entry->binding_count; b++) {
            fwrite(&entry->bindings[b], sizeof(gl_word_binding_t), 1, f);
        }
    }

    /* v5: no template block — emergent grammar via SNN bridge. */

    /* v2: dialect tag (#14). Always exactly GL_MAX_DIALECT_LEN bytes
     * for a fixed-width record so future fields can be appended without
     * a length prefix. */
    fwrite(gl->context_dialect, GL_MAX_DIALECT_LEN, 1, f);

    /* v3: compositional phrases (#9). count + per-phrase fixed record:
     *   form[GL_MAX_PHRASE_LEN] | component_words(u8) | frequency(u32)
     * semantic_vec is NOT persisted — it's a lazy cache rebuilt from
     * constituent words on first lookup post-load. */
    fwrite(&gl->phrase_count, sizeof(gl->phrase_count), 1, f);
    for (uint32_t pi = 0; pi < gl->phrase_count; pi++) {
        const gl_phrase_t* p = &gl->phrases[pi];
        fwrite(p->form, GL_MAX_PHRASE_LEN, 1, f);
        fwrite(&p->component_words, sizeof(p->component_words), 1, f);
        fwrite(&p->frequency, sizeof(p->frequency), 1, f);
    }

    fclose(f);
    LOG_INFO(LOG_MODULE,
             "Saved grounded language state to %s (%u words, %u phrases, dialect='%s')",
             path, gl->vocab_count, gl->phrase_count, gl->context_dialect);
    return 0;
}

grounded_language_t* grounded_language_load(const char* path, void* semantic_memory) {
    if (!path) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_load: NULL path");
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic, version, semantic_dim, vocab_count;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != GL_MAGIC) {
        fclose(f);
        return NULL;
    }
    fread_chk(&version, sizeof(version), 1, f);
    fread_chk(&semantic_dim, sizeof(semantic_dim), 1, f);
    fread_chk(&vocab_count, sizeof(vocab_count), 1, f);

    /* Sanity-bound header values — corrupt or truncated checkpoints
     * can otherwise drive grounded_language_create into multi-GiB
     * allocations or send the read loop into billions of iterations.
     * 4096 is a generous cap (current GL_SEMANTIC_DIM is 128); GL_MAX_VOCAB
     * is the hard ceiling for any legitimate file. */
    if (semantic_dim == 0 || semantic_dim > 4096 ||
        vocab_count > GL_MAX_VOCAB) {
        fclose(f);
        return NULL;
    }

    grounded_language_t* gl = grounded_language_create(semantic_dim, semantic_memory);
    if (!gl) { fclose(f); return NULL; }

    /* Read word entries */
    for (uint32_t w = 0; w < vocab_count; w++) {
        char form[GL_MAX_WORD_LEN];
        uint32_t frequency;
        gl_word_class_t word_class;
        float valence, arousal;
        bool ctx_init;

        fread_chk(form, GL_MAX_WORD_LEN, 1, f);
        fread_chk(&frequency, sizeof(frequency), 1, f);
        fread_chk(&word_class, sizeof(word_class), 1, f);
        fread_chk(&valence, sizeof(valence), 1, f);
        fread_chk(&arousal, sizeof(arousal), 1, f);

        gl_lexicon_entry_t* entry = lexicon_find_or_create(gl, form);
        if (!entry) continue;

        entry->frequency = frequency;
        entry->learned_class = word_class;
        entry->valence = valence;
        entry->arousal = arousal;

        fread_chk(&ctx_init, sizeof(ctx_init), 1, f);
        entry->context_initialized = ctx_init;
        if (ctx_init) {
            if (version >= 4) {
                /* v4 quantized read: max_abs + int8 codes. */
                float max_abs = 0.0f;
                fread_chk(&max_abs, sizeof(float), 1, f);
                if (max_abs > 0.0f) {
                    /* Heap-allocate so semantic_dim can grow without
                     * silently corrupting context_vectors. The header
                     * sanity-check above caps semantic_dim at 4096. */
                    int8_t* qbuf = (int8_t*)nimcp_calloc(semantic_dim,
                                                          sizeof(int8_t));
                    if (!qbuf) {
                        grounded_language_destroy(gl);
                        fclose(f);
                        return NULL;
                    }
                    /* calloc zero-initializes, so a short fread leaves
                     * residual zeros in qbuf rather than uninitialized
                     * stack memory bleeding into context_vector. */
                    size_t got = fread(qbuf, sizeof(int8_t), semantic_dim, f);
                    (void)got;
                    float scale = max_abs / 127.0f;
                    for (uint32_t d = 0; d < semantic_dim; d++) {
                        entry->context_vector[d] = (float)qbuf[d] * scale;
                    }
                    nimcp_free(qbuf);
                }
                /* max_abs == 0 → zero vector, nothing more on disk. */
            } else {
                /* v1-v3 legacy: float block. */
                fread_chk(entry->context_vector, sizeof(float),
                          semantic_dim, f);
            }
        }

        uint32_t binding_count;
        fread_chk(&binding_count, sizeof(binding_count), 1, f);

        for (uint32_t b = 0; b < binding_count; b++) {
            gl_word_binding_t binding;
            fread_chk(&binding, sizeof(binding), 1, f);

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

    /* Legacy template block (v<5). Templates were 8-slot fixed records:
     *   sizeof(gl_template_t) = sizeof(int)            // type enum
     *                         + 8 * sizeof(int)        // slots[8] enums
     *                         + sizeof(uint32_t)       // slot_count
     *                         + 2 * sizeof(float)      // frequency, confidence
     *                         = 4 + 32 + 4 + 8 = 48 bytes
     * The struct is removed in v5; we discard the bytes to advance the
     * file cursor to the dialect block. */
    if (version < 5) {
        uint32_t legacy_template_count = 0;
        fread_chk(&legacy_template_count, sizeof(legacy_template_count), 1, f);
        if (legacy_template_count > 0) {
            const long legacy_template_size = 48L;
            if (fseek(f, (long)legacy_template_count * legacy_template_size,
                      SEEK_CUR) != 0) {
                LOG_WARN(LOG_MODULE,
                         "grounded_language_load: failed to skip %u legacy templates",
                         legacy_template_count);
            }
        }
    }

    /* v2+: context_dialect (#14). v1 files don't have it; leave the
     * default empty string. fread on EOF returns 0 — silent fallback
     * preserves backwards compatibility. */
    if (version >= 2) {
        char dialect_buf[GL_MAX_DIALECT_LEN];
        size_t got = fread(dialect_buf, GL_MAX_DIALECT_LEN, 1, f);
        if (got == 1) {
            /* Force NUL termination defensively in case the file was
             * written without it. */
            dialect_buf[GL_MAX_DIALECT_LEN - 1] = '\0';
            memcpy(gl->context_dialect, dialect_buf, GL_MAX_DIALECT_LEN);
        }
    }

    /* v3+: compositional phrases (#9). v2 and earlier files don't have
     * this section; phrase_count stays 0 and phrases will be re-learned
     * from text. */
    if (version >= 3) {
        uint32_t saved_phrase_count = 0;
        if (fread(&saved_phrase_count, sizeof(saved_phrase_count), 1, f) == 1) {
            uint32_t cap = (saved_phrase_count > GL_MAX_PHRASES)
                ? GL_MAX_PHRASES : saved_phrase_count;
            for (uint32_t pi = 0; pi < cap; pi++) {
                char form_buf[GL_MAX_PHRASE_LEN];
                uint8_t comp_n;
                uint32_t freq;
                if (fread(form_buf, GL_MAX_PHRASE_LEN, 1, f) != 1) break;
                if (fread(&comp_n, sizeof(comp_n), 1, f) != 1) break;
                if (fread(&freq,  sizeof(freq),   1, f) != 1) break;
                form_buf[GL_MAX_PHRASE_LEN - 1] = '\0';
                gl_phrase_t* p = _gl_phrase_find_or_create(gl, form_buf, comp_n);
                if (p) p->frequency = freq;
            }
        }
    }

    fclose(f);
    LOG_INFO(LOG_MODULE,
             "Loaded grounded language state from %s (%u words, %u phrases, version=%u, dialect='%s')",
             path, gl->vocab_count, gl->phrase_count, version,
             gl->context_dialect);
    return gl;
}

/*=============================================================================
 * Internal — exposed via nimcp_grounded_language_internal.h
 *===========================================================================*/

gl_lexicon_entry_t* gl_internal_lexicon_find_or_create(
    grounded_language_t* gl,
    const char* word) {
    if (!gl || !word) return NULL;
    return lexicon_find_or_create(gl, word);
}
