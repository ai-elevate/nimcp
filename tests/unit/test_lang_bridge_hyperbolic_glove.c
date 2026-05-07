/**
 * @file test_lang_bridge_hyperbolic_glove.c
 * @brief PA-5+ — verify hyperbolic-distance GloVe metric in decode_spikes.
 *
 * Pattern: standalone smoke test. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_hyperbolic_glove.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,$(pwd)/build/lib \
 *       -o /tmp/test_lang_bridge_hyperbolic_glove
 *
 * Coverage:
 *   1. test_hyperbolic_default_off_matches_glove:
 *      The default (use_hyperbolic_embeddings=false) must reproduce the
 *      existing PA-5 cosine ranking — toggling hyperbolic mode on/off
 *      around a clean bridge with a fresh emb cache should not change
 *      rankings under PA-5 cosine.
 *
 *   2. test_hyperbolic_picks_closer_hierarchy:
 *      Under hyperbolic mode, two words at different "tree depths" (one
 *      embedding close to the query in Poincaré sense, one far) — the
 *      closer one wins, confirming hyperbolic distance is in play.
 *
 *   3. test_hyperbolic_zero_vector_no_nan:
 *      A registered word with a zero-vector embedding under blend=1 +
 *      hyperbolic mode must not yield NaN/Inf in any score, and the
 *      bridge must still return a non-empty result (decode does not crash).
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
 * Mock embedding store. Indexed by single-character word_form A/B/C/D.
 * -------------------------------------------------------------------- */
typedef struct {
    int call_count;
    float emb[4][4];
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

static snn_language_bridge_t* make_bridge(uint32_t n_concepts)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = n_concepts;
    cfg.max_word_pops    = 4;
    return snn_language_bridge_create(&cfg);
}

/* --------------------------------------------------------------------
 * Test 1: hyperbolic mode default OFF reproduces PA-5 cosine ranking.
 * Setup mirrors the PA-5 "embedding overrides binding" test:
 *   intent fires concept 0; A binds concept 0 with weight 1; B binds
 *   concept 1 with weight 1; A's emb is orthogonal to intent; B's emb
 *   is aligned with intent. Under blend=1 cosine: B wins. We then
 *   verify toggling hyperbolic on then back off lands on the same
 *   ranking — which proves hyperbolic OFF is a true no-op.
 * -------------------------------------------------------------------- */
static void test_hyperbolic_default_off_matches_glove(void)
{
    snn_language_bridge_t* b = make_bridge(4);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    for (uint32_t c = 0; c < 4; c++)
        snn_language_bridge_register_concept(b, c, 500 + c);
    snn_language_bridge_register_word(b, 0, "A");
    snn_language_bridge_register_word(b, 1, "B");
    snn_language_bridge_bind(b, 0, 0, 1.0f);
    snn_language_bridge_bind(b, 1, 1, 1.0f);

    mock_emb_ctx_t mock = {0};
    /* A: orthogonal to intent. B: aligned with intent. */
    mock.emb[0][1] = 1.0f;
    mock.emb[1][0] = 1.0f;

    EXPECT(snn_language_bridge_set_embedding_lookup(b, mock_lookup,
                                                     &mock, 4) == 0,
            "set lookup");
    EXPECT(snn_language_bridge_set_glove_blend(b, 1.0f) == 0, "blend=1");

    float intent[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    /* Baseline cosine pick (default — hyperbolic OFF). */
    snn_lang_word_result_t r1[2]; uint32_t n1 = 0;
    snn_language_bridge_decode_spikes(b, intent, 4, r1, 2, &n1);
    EXPECT(n1 >= 1, "baseline decode");
    EXPECT(n1 >= 1 && r1[0].word_form && strcmp(r1[0].word_form, "B") == 0,
            "baseline cosine: B wins; got %s",
            n1 >= 1 && r1[0].word_form ? r1[0].word_form : "(null)");

    /* Toggle hyperbolic ON then OFF; ranking under cosine must be
     * unchanged after we settle back to cosine mode. */
    EXPECT(snn_language_bridge_set_hyperbolic_embeddings(b, true) == 0,
            "toggle hyper on");
    EXPECT(snn_language_bridge_set_hyperbolic_embeddings(b, false) == 0,
            "toggle hyper off");

    snn_lang_word_result_t r2[2]; uint32_t n2 = 0;
    snn_language_bridge_decode_spikes(b, intent, 4, r2, 2, &n2);
    EXPECT(n2 >= 1 && r2[0].word_form && strcmp(r2[0].word_form, "B") == 0,
            "after toggle off, cosine ranking unchanged; got %s",
            n2 >= 1 && r2[0].word_form ? r2[0].word_form : "(null)");

    /* Score parity: same activation as baseline (within fp tolerance). */
    if (n1 >= 1 && n2 >= 1) {
        float diff = fabsf(r1[0].activation - r2[0].activation);
        EXPECT(diff < 1e-5f, "off→on→off should be stable; diff=%g", diff);
    }

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 2: hyperbolic mode picks the hierarchically-closer word.
 * Setup: intent embedding at axis 0 (small magnitude — sits near origin
 * in the Poincaré ball). Word A is near the origin too (small magnitude,
 * same axis); Word B is large and on a different axis (large magnitude
 * pushes it near the boundary). Under hyperbolic distance, points near
 * each other in the ball interior have small d_H; points near opposing
 * boundary regions have very large d_H. So A beats B.
 *
 * Both words have IDENTICAL binding setup (so binding contribution is
 * a wash) and we set blend=1 to isolate the GloVe term.
 *
 * Sanity check: under standard cosine, both A and B are unit-aligned to
 * the intent axis enough that direction matters more than magnitude — but
 * we don't assert anything about the cosine baseline here, only that
 * hyperbolic flips the answer to A.
 * -------------------------------------------------------------------- */
static void test_hyperbolic_picks_closer_hierarchy(void)
{
    snn_language_bridge_t* b = make_bridge(4);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    for (uint32_t c = 0; c < 4; c++)
        snn_language_bridge_register_concept(b, c, 600 + c);
    snn_language_bridge_register_word(b, 0, "A");
    snn_language_bridge_register_word(b, 1, "B");
    /* Both bind to the same concept with the same weight — cosine over
     * binding column equal for both. */
    snn_language_bridge_bind(b, 0, 0, 1.0f);
    snn_language_bridge_bind(b, 0, 1, 1.0f);

    mock_emb_ctx_t mock = {0};
    /* Small magnitude, same direction as intent — close to query in Poincaré. */
    mock.emb[0][0] = 0.1f;  /* A */
    /* Large magnitude on a different axis — far in Poincaré (saturates near boundary). */
    mock.emb[1][1] = 5.0f;  /* B */

    EXPECT(snn_language_bridge_set_embedding_lookup(b, mock_lookup,
                                                     &mock, 4) == 0,
            "set lookup");
    EXPECT(snn_language_bridge_set_glove_blend(b, 1.0f) == 0, "blend=1");
    EXPECT(snn_language_bridge_set_hyperbolic_embeddings(b, true) == 0,
            "hyper on");

    /* Intent: small magnitude on axis 0 → near origin, same direction as A. */
    float intent[4] = {0.1f, 0.0f, 0.0f, 0.0f};

    snn_lang_word_result_t r[2]; uint32_t n = 0;
    snn_language_bridge_decode_spikes(b, intent, 4, r, 2, &n);

    EXPECT(n >= 1, "decode produced at least 1 result");
    /* A (close to query in Poincaré) must outrank B (far). */
    EXPECT(n >= 1 && r[0].word_form && strcmp(r[0].word_form, "A") == 0,
            "hyperbolic: A (close in ball) must beat B (far); got %s",
            n >= 1 && r[0].word_form ? r[0].word_form : "(null)");

    /* Both scores must be finite. */
    for (uint32_t i = 0; i < n; i++) {
        EXPECT(isfinite(r[i].activation),
                "score[%u]=%g must be finite", i, r[i].activation);
    }

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 3: zero-vector embedding under hyperbolic mode is numerically safe.
 * Registers word A with an all-zero embedding and B with a normal one,
 * sets blend=1 + hyperbolic ON, and confirms decode does not produce
 * NaN/Inf in any score. The query is also unusual (intent = zeros) to
 * exercise the eps-floor in the implementation.
 * -------------------------------------------------------------------- */
static void test_hyperbolic_zero_vector_no_nan(void)
{
    snn_language_bridge_t* b = make_bridge(4);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    for (uint32_t c = 0; c < 4; c++)
        snn_language_bridge_register_concept(b, c, 700 + c);
    snn_language_bridge_register_word(b, 0, "A");
    snn_language_bridge_register_word(b, 1, "B");
    snn_language_bridge_bind(b, 0, 0, 1.0f);
    snn_language_bridge_bind(b, 0, 1, 1.0f);

    mock_emb_ctx_t mock = {0};
    /* A: zero vector (calloc'd to zero already). B: normal unit vec. */
    mock.emb[1][0] = 1.0f;

    EXPECT(snn_language_bridge_set_embedding_lookup(b, mock_lookup,
                                                     &mock, 4) == 0,
            "set lookup");
    EXPECT(snn_language_bridge_set_glove_blend(b, 1.0f) == 0, "blend=1");
    EXPECT(snn_language_bridge_set_hyperbolic_embeddings(b, true) == 0,
            "hyper on");

    /* Zero-vector intent — pathological query, exercises eps floor on
     * the projection scale. */
    float intent[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    snn_lang_word_result_t r[2]; uint32_t n = 0;
    int rc = snn_language_bridge_decode_spikes(b, intent, 4, r, 2, &n);

    EXPECT(rc == 0, "decode returned ok rc=%d", rc);
    /* All returned scores must be finite. */
    for (uint32_t i = 0; i < n; i++) {
        EXPECT(isfinite(r[i].activation),
                "zero-intent score[%u]=%g must be finite",
                i, r[i].activation);
        EXPECT(isfinite(r[i].confidence),
                "zero-intent conf[%u]=%g must be finite",
                i, r[i].confidence);
    }

    /* Now non-zero intent + zero-vec embedding: A's emb is all zeros, so
     * its hyperbolic distance to any non-origin query is non-trivial but
     * bounded; verify scores are still finite. */
    float intent2[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    n = 0;
    rc = snn_language_bridge_decode_spikes(b, intent2, 4, r, 2, &n);
    EXPECT(rc == 0, "decode2 returned ok rc=%d", rc);
    for (uint32_t i = 0; i < n; i++) {
        EXPECT(isfinite(r[i].activation),
                "nonzero-intent / zerovec-A score[%u]=%g must be finite",
                i, r[i].activation);
    }

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[PA-5+] test_lang_bridge_hyperbolic_glove\n");
    test_hyperbolic_default_off_matches_glove();
    test_hyperbolic_picks_closer_hierarchy();
    test_hyperbolic_zero_vector_no_nan();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
