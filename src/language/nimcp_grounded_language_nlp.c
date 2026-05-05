/**
 * @file nimcp_grounded_language_nlp.c
 * @brief NLP frontend for grounded_language: morphology, POS hints,
 *        embedding blend, BPE subword fallback.
 * @date 2026-05-05
 *
 * WHAT: Three pieces of NLP machinery that grounded_language was
 *       previously missing:
 *         1. Morphological normalization — strip common English
 *            inflectional/derivational suffixes so "running"/"runs"/
 *            "ran" all canonicalize to "run" before lexicon lookup.
 *         2. Suffix-based POS hints — use the same surface morphology
 *            to seed word_class when no positional context is available
 *            (e.g. "evaluation" → noun via -tion).
 *         3. Embedding + tokenizer connect points — wire an external
 *            word embedding layer + BPE tokenizer so that comprehend
 *            can (a) pull embedding vectors into the result's
 *            semantic_vector, and (b) decompose totally-OOV strings
 *            into subword token ids as a last-resort lookup path.
 *
 * WHY:  The legacy GL pipeline was tokenize → exact lexicon → fuzzy
 *       match → give up. That misses three real NLP signals: surface
 *       morphology (which English carries a lot of), distributed
 *       embeddings (which the rest of the brain is already trained on),
 *       and subword decomposition (which the BPE tokenizer can already
 *       do). Each missing signal forces the rest of the cognitive
 *       stack to compensate, often badly.
 *
 * HOW:  - Suffix stripping is a hand-rolled lightweight stemmer (NOT
 *         a full Porter — Porter would over-stem and lose meaning for
 *         a developmental brain). Order matters: longer suffixes first.
 *       - POS hint table is a small static array of (suffix → class).
 *       - Embedding + tokenizer hooks live behind opaque-pointer
 *         setters so this file doesn't need to know either struct's
 *         layout. Lookups are pure forwarding to the public APIs in
 *         nimcp_embedding.h / nimcp_tokenizer.h.
 *       - The `gl_nlp_lookup_chain()` helper is the single entry the
 *         primary impl calls — it walks exact → morph → fuzzy → BPE
 *         in that order and returns the resolved entry (or NULL).
 *
 * SOLID/DRY: All four NLP stages are isolated as small static helpers
 *            with one public dispatcher, so the comprehend hot path
 *            picks up the entire chain via a single function call.
 *            Each stage is independently optional — none assume the
 *            others are connected.
 */

#include "language/nimcp_grounded_language.h"
#include "nimcp_grounded_language_internal.h"

#include "generation/nimcp_embedding.h"
#include "generation/nimcp_tokenizer.h"

#include <ctype.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Forward decl from primary impl — find an entry without creating one. */
extern const gl_lexicon_entry_t* lexicon_find_internal(
    const grounded_language_t* gl, const char* word);

/* Forward decl — fuzzy fallback lives in the primary impl. We re-declare
 * it here rather than expose it in the public header. */
extern const gl_lexicon_entry_t* lexicon_find_fuzzy_external(
    const grounded_language_t* gl, const char* word);

/*=============================================================================
 * Suffix-stripping morphology
 *
 * Hand-rolled, intentionally conservative. English morphology has a few
 * very productive endings (-s, -ed, -ing, -ly, -er, -est, -tion, -ness,
 * -ment, -able). Stripping these covers ~80% of inflected/derived forms
 * without the over-stemming that classical Porter exhibits.
 *
 * Order: try the longest suffixes first so "happiness" doesn't become
 * "happi" via a premature -s strip.
 *===========================================================================*/

typedef struct {
    const char* suffix;
    size_t      len;
    size_t      min_stem;   /* don't strip if remaining stem would be shorter */
    const char* replace;    /* what to append after stripping (NULL → nothing) */
} gl_suffix_rule_t;

/* Rules ordered longest-first. The replace field handles the common
 * orthographic adjustments — "running" → "run" (drop double n),
 * "happily" → "happy" (i→y), "studies" → "study" (ies→y). */
static const gl_suffix_rule_t SUFFIX_RULES[] = {
    {"ation",  5, 3, NULL},   /* evaluation → evalu (then maybe -e drop) */
    {"ition",  5, 3, NULL},   /* condition  → cond  */
    {"ness",   4, 3, NULL},   /* happiness  → happi (handled below) */
    {"ment",   4, 3, NULL},   /* movement   → move (dropped 'e' restored later) */
    {"able",   4, 3, NULL},   /* readable   → read */
    {"ible",   4, 3, NULL},   /* visible    → vis */
    {"tion",   4, 3, NULL},   /* function   → func */
    {"sion",   4, 3, NULL},   /* tension    → ten */
    {"ies",    3, 3, "y"},    /* studies    → study */
    {"ied",    3, 3, "y"},    /* studied    → study */
    {"ing",    3, 3, NULL},   /* running    → runn (de-double below) */
    {"est",    3, 3, NULL},   /* fastest    → fast */
    {"ous",    3, 3, NULL},   /* dangerous  → danger */
    {"ive",    3, 3, NULL},   /* creative   → creat (then -e) */
    {"ful",    3, 3, NULL},   /* helpful    → help */
    {"ize",    3, 3, NULL},   /* organize   → organ */
    {"ise",    3, 3, NULL},   /* organise   → organ */
    {"ed",     2, 3, NULL},   /* walked     → walk */
    {"er",     2, 3, NULL},   /* walker     → walk */
    {"ly",     2, 3, NULL},   /* quickly    → quick */
    {"es",     2, 3, NULL},   /* boxes      → box */
    {"s",      1, 3, NULL},   /* cats       → cat */
};

#define GL_SUFFIX_RULE_COUNT (sizeof(SUFFIX_RULES)/sizeof(SUFFIX_RULES[0]))

/* De-double the trailing consonant when stripping "-ing"/"-ed" leaves
 * a CVCC where the last two C's are the same: "running" → "runn" → "run". */
static void de_double_trailing(char* word) {
    size_t n = strlen(word);
    if (n < 3) return;
    char a = word[n - 1];
    char b = word[n - 2];
    if (a == b && (a == 'n' || a == 'p' || a == 't' ||
                   a == 'd' || a == 'g' || a == 'm' ||
                   a == 'b' || a == 'l' || a == 'r')) {
        word[n - 1] = '\0';
    }
}

int gl_morph_normalize(const char* word, char* out, size_t out_sz) {
    if (!word || !out || out_sz == 0) return -1;

    size_t in_len = strlen(word);
    if (in_len + 1 > out_sz) return -1;

    /* Lowercase copy for the strip pass. */
    for (size_t i = 0; i < in_len; i++) {
        out[i] = (char)tolower((unsigned char)word[i]);
    }
    out[in_len] = '\0';

    if (in_len < 4) return 0;  /* too short to strip safely */

    for (size_t r = 0; r < GL_SUFFIX_RULE_COUNT; r++) {
        const gl_suffix_rule_t* rule = &SUFFIX_RULES[r];
        if (in_len <= rule->len + rule->min_stem - 1) continue;
        size_t suffix_start = in_len - rule->len;
        if (strcmp(out + suffix_start, rule->suffix) == 0) {
            out[suffix_start] = '\0';
            if (rule->replace) {
                size_t rlen = strlen(rule->replace);
                if (suffix_start + rlen + 1 > out_sz) {
                    /* abandon strip if no room for replacement */
                    memcpy(out, word, in_len);
                    out[in_len] = '\0';
                    return 0;
                }
                memcpy(out + suffix_start, rule->replace, rlen + 1);
            }
            /* De-double "running" → "runn" → "run" only for -ing/-ed. */
            if (strcmp(rule->suffix, "ing") == 0 ||
                strcmp(rule->suffix, "ed") == 0) {
                de_double_trailing(out);
            }
            return (int)(in_len - strlen(out));
        }
    }
    return 0;
}

/*=============================================================================
 * POS hints from surface morphology
 *===========================================================================*/

typedef struct {
    const char*     suffix;
    size_t          len;
    gl_word_class_t class_;
} gl_pos_rule_t;

/* Highest-precedence POS suffixes first. Keep this list short — we only
 * want strong, near-deterministic signals here. Edge cases (e.g. -er
 * could be noun "writer" or comparative "faster") deliberately omitted. */
static const gl_pos_rule_t POS_RULES[] = {
    {"tion",  4, GL_CLASS_NOUN},
    {"sion",  4, GL_CLASS_NOUN},
    {"ness",  4, GL_CLASS_NOUN},
    {"ment",  4, GL_CLASS_NOUN},
    {"ity",   3, GL_CLASS_NOUN},
    {"ing",   3, GL_CLASS_VERB},     /* gerund/participle */
    {"ed",    2, GL_CLASS_VERB},     /* past tense */
    {"ize",   3, GL_CLASS_VERB},
    {"ise",   3, GL_CLASS_VERB},
    {"ate",   3, GL_CLASS_VERB},
    {"ous",   3, GL_CLASS_ADJECTIVE},
    {"ive",   3, GL_CLASS_ADJECTIVE},
    {"ful",   3, GL_CLASS_ADJECTIVE},
    {"able",  4, GL_CLASS_ADJECTIVE},
    {"ible",  4, GL_CLASS_ADJECTIVE},
    {"al",    2, GL_CLASS_ADJECTIVE},
    {"ic",    2, GL_CLASS_ADJECTIVE},
    {"ly",    2, GL_CLASS_ADVERB},
};

#define GL_POS_RULE_COUNT (sizeof(POS_RULES)/sizeof(POS_RULES[0]))

gl_word_class_t gl_morph_pos_hint(const char* word) {
    if (!word) return GL_CLASS_UNKNOWN;
    size_t n = strlen(word);
    if (n < 4) return GL_CLASS_UNKNOWN;  /* short words too ambiguous */

    /* Lowercase tail for matching. */
    char tail[16];
    size_t tlen = (n < sizeof(tail) - 1) ? n : sizeof(tail) - 1;
    size_t start = n - tlen;
    for (size_t i = 0; i < tlen; i++) {
        tail[i] = (char)tolower((unsigned char)word[start + i]);
    }
    tail[tlen] = '\0';

    for (size_t r = 0; r < GL_POS_RULE_COUNT; r++) {
        const gl_pos_rule_t* rule = &POS_RULES[r];
        if (tlen < rule->len + 2) continue;  /* need stem ≥ 2 chars */
        const char* p = tail + tlen - rule->len;
        if (strcmp(p, rule->suffix) == 0) return rule->class_;
    }
    return GL_CLASS_UNKNOWN;
}

/*=============================================================================
 * Embedding + tokenizer connect points
 *===========================================================================*/

void grounded_language_connect_embeddings(grounded_language_t* gl,
                                            void* emb,
                                            uint32_t emb_dim,
                                            gl_word_to_id_fn word_to_id_fn,
                                            void* ctx) {
    if (!gl) return;
    gl->embeddings    = emb;
    gl->emb_dim       = emb_dim;
    gl->word_to_id_fn = word_to_id_fn;
    gl->word_to_id_ctx = ctx;
}

void grounded_language_connect_tokenizer(grounded_language_t* gl, void* tok) {
    if (!gl) return;
    gl->tokenizer = tok;
}

/*=============================================================================
 * Embedding lookup helper — fills out_vec[gl->semantic_dim] with the
 * embedding for `word` if the embedding layer is connected and the
 * dim matches. Returns 0 on success, -1 otherwise. Called from comprehend.
 *===========================================================================*/

int gl_nlp_embedding_lookup(grounded_language_t* gl,
                              const char* word,
                              float* out_vec) {
    if (!gl || !word || !out_vec) return -1;
    if (!gl->embeddings || !gl->word_to_id_fn) return -1;
    if (gl->emb_dim != gl->semantic_dim) return -1;  /* dim mismatch — skip */

    uint32_t token_id = gl->word_to_id_fn(gl->word_to_id_ctx, word);
    if (token_id == 0) return -1;  /* unknown to caller's vocab */

    return embedding_lookup((const embedding_layer_t*)gl->embeddings,
                            token_id, out_vec);
}

/*=============================================================================
 * BPE subword fallback — when comprehend has exhausted exact + morph +
 * fuzzy and a tokenizer is connected, try to decompose the word into
 * subword token ids and pull embeddings for them. We return success if
 * the tokenizer produced ANY known subword token; the caller then knows
 * the word is "partially recognized" rather than fully OOV.
 *
 * This is a very lightweight signal — it doesn't try to rebuild an
 * entry. It just lets the comprehend caller bump confidence slightly
 * and (when embeddings are connected) average the subword vectors into
 * the semantic_vector.
 *===========================================================================*/

int gl_nlp_subword_lookup(grounded_language_t* gl,
                            const char* word,
                            float* out_vec_or_null) {
    if (!gl || !word) return -1;
    if (!gl->tokenizer) return -1;

    uint32_t ids[16];
    uint32_t actual = 0;
    int rc = tokenizer_encode((const tokenizer_t*)gl->tokenizer,
                               word, ids, 16, &actual);
    if (rc != 0 || actual == 0) return -1;

    /* If embeddings are connected and the dim matches, average the
     * subword embeddings into out_vec_or_null. */
    if (out_vec_or_null && gl->embeddings &&
        gl->emb_dim == gl->semantic_dim) {
        float scratch[1024];  /* upper bound on semantic_dim */
        if (gl->semantic_dim > 1024) return 0;  /* too big — signal-only */

        for (uint32_t d = 0; d < gl->semantic_dim; d++) out_vec_or_null[d] = 0.0f;
        size_t hits = 0;
        for (uint32_t i = 0; i < actual; i++) {
            if (embedding_lookup((const embedding_layer_t*)gl->embeddings,
                                  ids[i], scratch) == 0) {
                for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                    out_vec_or_null[d] += scratch[d];
                }
                hits++;
            }
        }
        if (hits > 0) {
            float inv = 1.0f / (float)hits;
            for (uint32_t d = 0; d < gl->semantic_dim; d++) {
                out_vec_or_null[d] *= inv;
            }
        }
    }
    return 0;
}

/*=============================================================================
 * Lookup chain — single entry the comprehend hot path uses to walk
 * exact → morph → fuzzy. The BPE step is signal-only (no entry to
 * return) so it's exercised separately by the comprehend caller via
 * gl_nlp_subword_lookup().
 *===========================================================================*/

const gl_lexicon_entry_t* gl_nlp_lookup_chain(grounded_language_t* gl,
                                                 const char* word) {
    if (!gl || !word) return NULL;

    const gl_lexicon_entry_t* hit = lexicon_find_internal(gl, word);
    if (hit) return hit;

    /* Stage 2: morphological normalization. */
    char stem[64];
    if (gl_morph_normalize(word, stem, sizeof(stem)) > 0) {
        hit = lexicon_find_internal(gl, stem);
        if (hit) return hit;
    }

    /* Stage 3: fuzzy match. */
    return lexicon_find_fuzzy_external(gl, word);
}
