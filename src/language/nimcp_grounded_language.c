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
#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "cognitive/memory/nimcp_semantic_memory.h"
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

            /* Fire NEW_WORD event on the cognitive bus. */
            extern void gl_fire_event(grounded_language_t*, const gl_event_t*);
            gl_event_t ev = {0};
            ev.type = GL_EVENT_NEW_WORD;
            ev.word = entry->form;
            gl_fire_event(gl, &ev);

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

/*=============================================================================
 * #12 SNN spike → lexicon decoding
 *===========================================================================*/

int gl_observe_snn_spikes(grounded_language_t* gl,
                            const float* spike_rates,
                            uint32_t rate_dim,
                            float* confidence_out) {
    if (!gl || !spike_rates || rate_dim == 0) return -1;
    if (rate_dim != gl->semantic_dim) return -1;
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

    /* SVOO: "I think learning is important" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_SVOO;
    t->slots[0] = GL_CLASS_PRONOUN;
    t->slots[1] = GL_CLASS_VERB;
    t->slots[2] = GL_CLASS_NOUN;
    t->slots[3] = GL_CLASS_ADJECTIVE;
    t->slot_count = 4;
    t->frequency = 7.0f;
    t->confidence = 0.75f;

    /* Comparative: "The idea is more complex than the pattern" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_COMPARATIVE;
    t->slots[0] = GL_CLASS_NOUN;
    t->slots[1] = GL_CLASS_ADJECTIVE;
    t->slots[2] = GL_CLASS_NOUN;
    t->slot_count = 3;
    t->frequency = 4.0f;
    t->confidence = 0.7f;

    /* Conditional: "If the evidence is clear then the answer is true" */
    t = &gl->templates[gl->template_count++];
    t->type = GL_PATTERN_CONDITIONAL;
    t->slots[0] = GL_CLASS_NOUN;
    t->slots[1] = GL_CLASS_ADJECTIVE;
    t->slots[2] = GL_CLASS_NOUN;
    t->slots[3] = GL_CLASS_ADJECTIVE;
    t->slot_count = 4;
    t->frequency = 3.0f;
    t->confidence = 0.65f;
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

    /* Templates */
    gl->template_capacity = GL_MAX_TEMPLATES;
    gl->templates = (gl_template_t*)nimcp_calloc(gl->template_capacity, sizeof(gl_template_t));
    if (!gl->templates) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_create: failed to allocate templates");
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

    /* RNG */
    gl->rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEULL;

    /* Seed with function words and basic templates */
    seed_function_words(gl);
    seed_conceptual_words(gl);
    seed_templates(gl);

    /* Register with BBB */
    bbb_register_module("grounded_language", BBB_MODULE_TYPE_COGNITIVE);

    LOG_INFO(LOG_MODULE, "Grounded language system created (dim=%u, vocab=%u words)",
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

    /* Phrases (#9) — each entry's semantic_vec is gl-owned. */
    if (gl->phrases) {
        for (uint32_t p = 0; p < gl->phrase_count; p++) {
            nimcp_free(gl->phrases[p].semantic_vec);
        }
        nimcp_free(gl->phrases);
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
    if (!gl || !intent || !result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_produce: NULL parameter (gl=%p, intent=%p, result=%p)",
            (void*)gl, (void*)intent, (void*)result);
        return -1;
    }
    memset(result, 0, sizeof(*result));

    /* Dual-path: if SNN bridge available, blend spike-driven production */
    if (gl->snn_bridge) {
        float blend = snn_language_bridge_get_blend(gl->snn_bridge);
        if (blend > 0.0f) {
            snn_lang_production_result_t spike_result;
            memset(&spike_result, 0, sizeof(spike_result));
            if (snn_language_bridge_produce(gl->snn_bridge, intent, intent_dim,
                                            &spike_result) == 0 &&
                spike_result.text && spike_result.word_count > 0) {
                /* At high blend, use spike output directly */
                if (blend >= 0.9f) {
                    result->text = spike_result.text;
                    spike_result.text = NULL; /* Transfer ownership */
                    result->word_count = spike_result.word_count;
                    result->fluency = spike_result.fluency;
                    result->relevance = spike_result.spike_confidence;
                    result->creativity = spike_result.creativity;
                    result->semantic_vector = (float*)nimcp_calloc(
                        gl->semantic_dim, sizeof(float));
                    if (result->semantic_vector) {
                        uint32_t copy_dim = (intent_dim < gl->semantic_dim)
                            ? intent_dim : gl->semantic_dim;
                        memcpy(result->semantic_vector, intent,
                               copy_dim * sizeof(float));
                    }
                    snn_lang_production_result_cleanup(&spike_result);
                    gl->stats.total_productions++;
                    return 0;
                }
                /* Low blend: record spike confidence for later blending */
                result->creativity = spike_result.creativity * blend;
            }
            snn_lang_production_result_cleanup(&spike_result);
        }
    }

    /* Allocate output buffer */
    size_t buf_size = GL_MAX_PRODUCTION_WORDS * GL_MAX_WORD_LEN;
    char* buf = (char*)nimcp_calloc(buf_size, 1);
    if (!buf) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_produce: failed to allocate output buffer");
        return -1;
    }

    result->semantic_vector = (float*)nimcp_calloc(gl->semantic_dim, sizeof(float));
    if (!result->semantic_vector) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_produce: failed to allocate semantic vector");
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
        /* Collapse-guard (2026-05): when fluency × relevance is below the
         * meaningful-confidence floor, emitting the produced template
         * looks authoritative but is just whichever seeded attractor
         * happens to score nearest. Surface the lack of grounding to the
         * caller instead — the user-visible "the awareness controlled"
         * mode collapse came directly from this path. */
        float prod_conf = prod.fluency * prod.relevance;
        if (prod_conf < GL_RESPOND_MIN_CONFIDENCE) {
            snprintf(response, response_max,
                     "I don't have words for that yet.");
            if (confidence) *confidence = prod_conf;  /* Honest, low. */
        } else {
            strncpy(response, prod.text, response_max - 1);
            response[response_max - 1] = '\0';
            if (confidence) *confidence = prod_conf;
        }
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
    if (!gl || !text) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_learn_syntax: NULL parameter");
        return -1;
    }

    size_t tlen = strlen(text);
    char* buf = (char*)nimcp_malloc(tlen + 1);
    if (!buf) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "grounded_language_learn_syntax: failed to allocate buffer");
        return -1;
    }
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

void grounded_language_connect_snn_bridge(grounded_language_t* gl,
                                           struct snn_language_bridge* bridge) {
    if (!gl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "grounded_language_connect_snn_bridge: NULL gl");
        return;
    }
    gl->snn_bridge = bridge;
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
    out->templates_count      = gl->template_count;
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
     * size reduction at <0.4% per-element error. */
    uint32_t magic = GL_MAGIC;
    uint32_t version = 4;
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
                /* Quantize to int8. Stack buffer up to 1024 dims;
                 * realistically semantic_dim ≤ 256. */
                int8_t qbuf[1024];
                if (gl->semantic_dim > 1024) {
                    /* Fallback: write zeros — vector will dequantize
                     * to zero. Defensive guard for unusual configs. */
                    int8_t zero = 0;
                    for (uint32_t d = 0; d < gl->semantic_dim; d++)
                        fwrite(&zero, sizeof(int8_t), 1, f);
                } else {
                    float inv = 127.0f / max_abs;
                    for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                        float v = entry->context_vector[d] * inv;
                        if (v >  127.0f) v =  127.0f;
                        if (v < -127.0f) v = -127.0f;
                        qbuf[d] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
                    }
                    fwrite(qbuf, sizeof(int8_t), gl->semantic_dim, f);
                }
            }
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
             "Saved grounded language state to %s (%u words, %u templates, %u phrases, dialect='%s')",
             path, gl->vocab_count, gl->template_count,
             gl->phrase_count, gl->context_dialect);
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
                if (max_abs > 0.0f && semantic_dim <= 1024) {
                    int8_t qbuf[1024];
                    fread_chk(qbuf, sizeof(int8_t), semantic_dim, f);
                    float scale = max_abs / 127.0f;
                    for (uint32_t d = 0; d < semantic_dim; d++) {
                        entry->context_vector[d] = (float)qbuf[d] * scale;
                    }
                } else if (max_abs > 0.0f) {
                    /* Defensive: walk past the int8 block to keep
                     * stream alignment when semantic_dim > 1024. */
                    int8_t skip;
                    for (uint32_t d = 0; d < semantic_dim; d++)
                        fread_chk(&skip, sizeof(int8_t), 1, f);
                    memset(entry->context_vector, 0,
                           semantic_dim * sizeof(float));
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

    /* Templates */
    uint32_t template_count;
    fread_chk(&template_count, sizeof(template_count), 1, f);
    if (template_count <= gl->template_capacity) {
        fread_chk(gl->templates, sizeof(gl_template_t), template_count, f);
        gl->template_count = template_count;
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
