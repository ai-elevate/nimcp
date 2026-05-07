/**
 * @file test_lang_bridge_glove_blend.c
 * @brief PA-5 — verify GloVe-aware decode blend in bridge_produce.
 *
 * Pattern: standalone smoke test. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_glove_blend.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_bridge_glove_blend
 *
 * Coverage:
 *   1. test_glove_blend_zero_matches_binding_only:
 *      Set glove_blend=0 (the default). Output ranking must equal pure
 *      Patch-A binding-only behavior — embedding lookup is bypassed.
 *
 *   2. test_glove_blend_one_overrides_binding:
 *      Word A bound to many concepts (high binding score) with embedding
 *      pointing away from intent. Word B bound to a single concept (low
 *      binding score) with embedding aligned to intent. With blend=1,
 *      B must win — embedding direction overrides binding density.
 *
 *   3. test_emb_cache_called_once_per_word:
 *      Counter-tracking callback. After 5 produce calls over the same
 *      4-word vocabulary, the callback was invoked at most 4 times
 *      (once per word) — cache hit on subsequent decodes.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static int g_failures = 0;

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d " #cond " : ", __func__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        g_failures++; \
    } \
} while (0)

/* --------------------------------------------------------------------
 * Mock embedding store. Indexed by single-character word_form; returns a
 * pre-baked unit vector. Tracks call count for cache verification.
 * -------------------------------------------------------------------- */
typedef struct {
    int call_count;
    /* Per-letter embeddings of dim 4. Caller asks for dim=4. */
    float emb[4][4];   /* A=row0, B=row1, C=row2, D=row3 */
} mock_emb_ctx_t;

static int mock_lookup(void* ctx, const char* word_form,
                        float* out_vec, uint32_t out_dim)
{
    mock_emb_ctx_t* m = (mock_emb_ctx_t*)ctx;
    m->call_count++;
    if (!word_form || !word_form[0] || word_form[1]) return -1;
    if (out_dim != 4) return -1;
    int idx;
    switch (word_form[0]) {
        case 'A': idx = 0; break;
        case 'B': idx = 1; break;
        case 'C': idx = 2; break;
        case 'D': idx = 3; break;
        default:  return -1;
    }
    memcpy(out_vec, m->emb[idx], sizeof(float) * 4);
    return 0;
}

/* Build a bridge with 4 words. Each test customizes bindings + embeddings. */
static snn_language_bridge_t* make_bridge(uint32_t n_concepts)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = n_concepts;
    cfg.max_word_pops    = 4;
    return snn_language_bridge_create(&cfg);
}

static void register_word(snn_language_bridge_t* b, uint32_t pop, const char* form)
{
    snn_language_bridge_register_word(b, pop, form);
}

static void register_concept(snn_language_bridge_t* b, uint32_t pop, uint64_t id)
{
    snn_language_bridge_register_concept(b, pop, id);
}

static void first_word(const char* text, char* out, size_t out_max)
{
    out[0] = '\0';
    if (!text) return;
    size_t i = 0;
    while (text[i] && text[i] != ' ' && i < out_max - 1) {
        out[i] = text[i];
        i++;
    }
    out[i] = '\0';
}

/* --------------------------------------------------------------------
 * Test 1: glove_blend = 0 disables embedding lookups. Verify by checking
 * the callback is NEVER called. We rig the same setup as PA-1's first
 * test: A bound to concept 0 strongly, B bound to 100 concepts diffusely.
 * Spike concept 0 → A wins (binding cosine). With blend=0, the embedding
 * callback should be skipped entirely.
 * -------------------------------------------------------------------- */
static void test_glove_blend_zero_matches_binding_only(void)
{
    snn_language_bridge_t* b = make_bridge(100);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    register_concept(b, 0, 100);
    register_word(b, 0, "A");
    register_word(b, 1, "B");
    snn_language_bridge_bind(b, 0, 0, 1.0f);  /* A: 1 binding, weight 1.0 */
    /* B: 1 binding to a *different* concept, also weight 1.0 — make sure
     * binding-only ranking still picks A when concept 0 is the only spike. */
    register_concept(b, 1, 101);
    snn_language_bridge_bind(b, 1, 1, 1.0f);

    mock_emb_ctx_t mock = {0};
    /* Embeddings present but irrelevant — blend=0 should not call the cb. */
    mock.emb[0][0] = 1.0f;  /* A */
    mock.emb[1][0] = 1.0f;  /* B (same as A; if used, tie) */
    EXPECT(snn_language_bridge_set_embedding_lookup(b, mock_lookup,
                                                     &mock, 4) == 0,
            "set_embedding_lookup");
    EXPECT(snn_language_bridge_set_glove_blend(b, 0.0f) == 0,
            "set_glove_blend(0)");

    float intent[100] = {0};
    intent[0] = 1.0f;
    snn_lang_word_result_t results[2];
    uint32_t n_out = 0;
    snn_language_bridge_decode_spikes(b, intent, 100, results, 2, &n_out);

    EXPECT(n_out >= 1, "got %u results", n_out);
    EXPECT(results[0].word_form && strcmp(results[0].word_form, "A") == 0,
            "blend=0: binding-only must pick A; got %s",
            results[0].word_form ? results[0].word_form : "(null)");
    EXPECT(mock.call_count == 0,
            "blend=0 must skip embedding lookup; got %d calls",
            mock.call_count);

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 2: glove_blend = 1 → embedding cosine dominates over binding-only.
 * Setup: intent fires concept 0 only.
 *   - Word A bound to concept 0 with weight 1.0  → binding cosine = 1.0
 *     Embedding orthogonal to intent              → glove cosine = 0
 *   - Word B bound to concept 1 with weight 1.0  → binding cosine = 0
 *     Embedding aligned with intent               → glove cosine = 1.0
 * blend=0: A wins (binding direction matches intent at concept 0).
 * blend=1: B wins (embedding direction matches intent's first coord).
 * -------------------------------------------------------------------- */
static void test_glove_blend_one_overrides_binding(void)
{
    snn_language_bridge_t* b = make_bridge(4);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    for (uint32_t c = 0; c < 4; c++) register_concept(b, c, 200 + c);
    register_word(b, 0, "A");
    register_word(b, 1, "B");

    /* A binds concept 0 → binding-only direction-matches intent. */
    snn_language_bridge_bind(b, 0, 0, 1.0f);
    /* B binds concept 1 → binding-only orthogonal to intent. */
    snn_language_bridge_bind(b, 1, 1, 1.0f);

    /* Intent spikes concept 0. concept_rates[0:4] doubles as the embedding
     * query because emb_dim == 4 and decode_spikes uses concept_rates[0:emb_dim]
     * as the embedding side of the cosine. */
    float intent[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    mock_emb_ctx_t mock = {0};
    /* A's embedding orthogonal to intent (axis 1 instead of 0). */
    mock.emb[0][0] = 0.0f; mock.emb[0][1] = 1.0f;
    /* B's embedding aligned with intent (axis 0). */
    mock.emb[1][0] = 1.0f; mock.emb[1][1] = 0.0f;

    EXPECT(snn_language_bridge_set_embedding_lookup(b, mock_lookup,
                                                     &mock, 4) == 0,
            "set_embedding_lookup");

    /* Sanity: blend=0 picks A — binding direction matches intent. */
    EXPECT(snn_language_bridge_set_glove_blend(b, 0.0f) == 0, "blend=0");
    snn_lang_word_result_t r[2]; uint32_t n;
    snn_language_bridge_decode_spikes(b, intent, 4, r, 2, &n);
    EXPECT(n >= 1 && r[0].word_form && strcmp(r[0].word_form, "A") == 0,
            "blend=0: A (binding-aligned) should win; got %s",
            n >= 1 && r[0].word_form ? r[0].word_form : "(null)");

    /* Now blend=1 — embedding alignment must override binding direction. */
    EXPECT(snn_language_bridge_set_glove_blend(b, 1.0f) == 0, "blend=1");
    snn_language_bridge_decode_spikes(b, intent, 4, r, 2, &n);
    EXPECT(n >= 1 && r[0].word_form && strcmp(r[0].word_form, "B") == 0,
            "blend=1: B (embedding-aligned) must win; got %s",
            n >= 1 && r[0].word_form ? r[0].word_form : "(null)");

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 3: per-word embedding cache — callback called ≤ N_words times
 * across many decodes.
 * -------------------------------------------------------------------- */
static void test_emb_cache_called_once_per_word(void)
{
    snn_language_bridge_t* b = make_bridge(4);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    for (uint32_t c = 0; c < 4; c++) register_concept(b, c, 300 + c);
    register_word(b, 0, "A");
    register_word(b, 1, "B");
    register_word(b, 2, "C");
    register_word(b, 3, "D");
    for (uint32_t i = 0; i < 4; i++) snn_language_bridge_bind(b, i, i, 1.0f);

    mock_emb_ctx_t mock = {0};
    for (uint32_t i = 0; i < 4; i++) mock.emb[i][i] = 1.0f;  /* unit basis */

    EXPECT(snn_language_bridge_set_embedding_lookup(b, mock_lookup,
                                                     &mock, 4) == 0,
            "set_embedding_lookup");
    EXPECT(snn_language_bridge_set_glove_blend(b, 0.5f) == 0, "blend=0.5");

    /* Run 5 decodes — same vocabulary, same registered words. After the
     * first decode all embeddings should be cached, so 4 callback calls
     * total (one per registered word). */
    float intent[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 5; i++) {
        snn_lang_word_result_t r[4]; uint32_t n;
        snn_language_bridge_decode_spikes(b, intent, 4, r, 4, &n);
    }
    EXPECT(mock.call_count == 4,
            "expected 4 cache fills (one per word); got %d", mock.call_count);

    /* Invalidate and verify the next decode re-queries. */
    snn_language_bridge_invalidate_emb_cache(b);
    snn_lang_word_result_t r[4]; uint32_t n;
    snn_language_bridge_decode_spikes(b, intent, 4, r, 4, &n);
    EXPECT(mock.call_count == 8,
            "expected 4 more lookups after invalidate (8 total); got %d",
            mock.call_count);

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[PA-5] test_lang_bridge_glove_blend\n");
    test_glove_blend_zero_matches_binding_only();
    test_glove_blend_one_overrides_binding();
    test_emb_cache_called_once_per_word();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
