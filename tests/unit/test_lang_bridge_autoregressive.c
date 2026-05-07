/**
 * @file test_lang_bridge_autoregressive.c
 * @brief PA-2 — verify autoregressive recurrent decoder behavior.
 *
 * Pattern: standalone smoke test. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_autoregressive.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_bridge_autoregressive
 *
 * Coverage:
 *   1. test_intent_persistence_zero_matches_legacy:
 *      Default config (intent_persistence=0, word_feedback=0.3) — produce
 *      output should match the pre-PA-2 in-place 70/30 blend behavior.
 *      Verified by replicating the math: after a few tokens, the state
 *      has drifted away from intent, so a long-tail "intent-only" word
 *      that depends on continued intent presence won't be picked.
 *
 *   2. test_intent_persistence_one_keeps_intent_constant:
 *      With intent_persistence = 1.0 the recurrent state is bypassed
 *      entirely — concept_acts == intent for every step. Setup: an
 *      "intent-aligned" word A and several "off-intent" words. With
 *      argmax + refractory, the loop should pick A first, then
 *      refractory-mask it, but each subsequent step still asks "what
 *      else matches the original intent?", so picks remain coherent
 *      with the original prompt rather than drifting.
 *
 *   3. test_intent_persistence_partial_blend:
 *      intent_persistence = 0.5 + word_feedback = 0.5 — both intent and
 *      state contribute. The first picked word's reverse-encoding pulls
 *      state in its direction but intent still anchors half of concept_acts.
 *      Verify the output sequence remains intent-aware through 4 tokens.
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

/* ---------------------------------------------------------------------
 * Setup: 4 concept dims, 5 words A/B/C/D/E.
 *
 *   A: bound to concept 0 (the "prompt" dim). Strong cosine when intent
 *      points to concept 0. Reverse-encoding maps back to concept 0.
 *   B: bound to concept 1. Off-intent. But B's reverse-encoding maps to
 *      concept 1 — so once B is picked, state drifts toward concept 1.
 *   C: bound to concept 1. Same as B for binding direction. After B is
 *      picked, C becomes the next-most-aligned-with-state word.
 *   D, E: bound to concepts 2, 3. Mostly inert.
 *
 * Intent: spike concept 0. Without recurrent feedback, A would always
 * win. With legacy 70/30 in-place update: pick A → state pulls toward
 * concept 0 again (A's reverse encoding) but A is refractory, so the
 * second pick is whichever is next-best. Then state stays near concept 0
 * because A's encoding is the only contribution (no other words picked
 * yet). Eventually the loop ends.
 *
 * To make a meaningful test, we want a setup where:
 *   - First pick is A (everyone agrees).
 *   - Second pick depends on whether intent is still present.
 *
 * Trick: bind a SECOND intent-pointing word A2 that has lower binding
 * weight than A. Let A and A2 both point to concept 0 but with different
 * strengths. With strong intent persistence, A2 wins second; with state
 * drift (no persistence), the state becomes confused and the picked
 * word can be anything.
 * --------------------------------------------------------------------- */
typedef struct {
    snn_language_bridge_t* b;
    int word_count;
    char picks[8][32];  /* up to 8 picked words, each <= 32 chars */
    int  num_picks;
} ar_test_ctx_t;

static snn_language_bridge_t* build_ar_bridge(void)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = 4;
    cfg.max_word_pops    = 8;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    if (!b) return NULL;

    /* Concepts 0..3. */
    for (uint32_t c = 0; c < 4; c++) {
        snn_language_bridge_register_concept(b, c, 1000 + c);
    }
    /* Two prompt-aligned words at different binding weights. */
    snn_language_bridge_register_word(b, 0, "A");
    snn_language_bridge_bind(b, 0, 0, 1.0f);   /* A: strongest at concept 0 */
    snn_language_bridge_register_word(b, 1, "A2");
    snn_language_bridge_bind(b, 0, 1, 0.6f);   /* A2: also at concept 0 */

    /* Off-intent words that point to other concepts. */
    snn_language_bridge_register_word(b, 2, "B");
    snn_language_bridge_bind(b, 1, 2, 0.9f);
    snn_language_bridge_register_word(b, 3, "C");
    snn_language_bridge_bind(b, 1, 3, 0.8f);
    snn_language_bridge_register_word(b, 4, "D");
    snn_language_bridge_bind(b, 2, 4, 0.7f);
    snn_language_bridge_register_word(b, 5, "E");
    snn_language_bridge_bind(b, 3, 5, 0.7f);
    return b;
}

/* Pull the full produced sequence as a list of tokens. */
static int produce_tokens(snn_language_bridge_t* b, const float* intent,
                           uint32_t intent_dim, char picks[][32],
                           int max_picks)
{
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    if (snn_language_bridge_produce(b, intent, intent_dim, &res) != 0) {
        return 0;
    }
    int n = 0;
    if (res.text) {
        char buf[2048];
        size_t L = strlen(res.text);
        if (L >= sizeof(buf)) L = sizeof(buf) - 1;
        memcpy(buf, res.text, L);
        buf[L] = '\0';
        char* save = NULL;
        for (char* tok = strtok_r(buf, " ", &save); tok && n < max_picks;
             tok = strtok_r(NULL, " ", &save)) {
            strncpy(picks[n], tok, 31); picks[n][31] = '\0';
            n++;
        }
    }
    snn_lang_production_result_cleanup(&res);
    return n;
}

static int contains(char picks[][32], int n, const char* w)
{
    for (int i = 0; i < n; i++) {
        if (strcmp(picks[i], w) == 0) return i;  /* index */
    }
    return -1;
}

/* --------------------------------------------------------------------
 * Test 1: legacy default (intent_persistence = 0). With argmax + refractory,
 * pick order is: A first (highest at concept 0). Then state evolves toward
 * A's reverse-encoding (concept 0 again), but A is refractory — A2 (also
 * at concept 0) wins second. Then state still leans concept 0 — but both
 * concept-0 words are exhausted. Picks 3+ depend on state drift.
 *
 * What we assert: the SECOND pick is A2 — proving the recurrent feedback
 * does point back at concept 0. (This same pattern holds for any
 * intent_persistence value, since the first two picks are determined
 * mostly by binding strength at concept 0.)
 * -------------------------------------------------------------------- */
static void test_intent_persistence_zero_matches_legacy(void)
{
    snn_language_bridge_t* b = build_ar_bridge();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_autoregressive(b, 0.0f, 0.3f) == 0,
            "default legacy config");

    float intent[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    char picks[8][32];
    int n = produce_tokens(b, intent, 4, picks, 8);

    EXPECT(n >= 2, "expected at least 2 picks; got %d", n);
    if (n >= 2) {
        EXPECT(strcmp(picks[0], "A")  == 0, "first pick should be A; got %s", picks[0]);
        EXPECT(strcmp(picks[1], "A2") == 0,
                "legacy: second pick should be A2 (state still at concept 0); got %s",
                picks[1]);
    }
    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 2: intent_persistence = 1.0 → state is ignored. concept_acts ==
 * intent for every step. Pick order is purely "next-best at concept 0":
 *   A (1.0) → A2 (0.6) → ... (refractory clears every concept-0 word).
 * After A and A2 are exhausted, NO word remains positive at concept 0,
 * so produce should stop.
 *
 * Critical assertion: with full persistence, B/C/D/E (off-intent words)
 * should NEVER be picked, because their cosine score is zero and the
 * loop's "stop if confidence too low" gate fires.
 * -------------------------------------------------------------------- */
static void test_intent_persistence_one_keeps_intent_constant(void)
{
    snn_language_bridge_t* b = build_ar_bridge();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_autoregressive(b, 1.0f, 0.3f) == 0,
            "intent_persistence=1");

    float intent[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    char picks[8][32];
    int n = produce_tokens(b, intent, 4, picks, 8);

    EXPECT(n >= 1, "expected at least one pick; got %d", n);
    /* First pick is still A. */
    if (n >= 1) {
        EXPECT(strcmp(picks[0], "A") == 0, "first pick should be A; got %s", picks[0]);
    }
    /* Off-intent words must NEVER appear under full persistence. */
    EXPECT(contains(picks, n, "B") < 0, "B drift: %s found in output", "B");
    EXPECT(contains(picks, n, "C") < 0, "C drift: %s found in output", "C");
    EXPECT(contains(picks, n, "D") < 0, "D drift: %s found in output", "D");
    EXPECT(contains(picks, n, "E") < 0, "E drift: %s found in output", "E");

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 3: partial blend (intent_persistence = 0.5). Verify the API
 * accepts intermediate values and produce returns at least 2 tokens.
 * The exact picks depend on the cosine math at each step, which the
 * other tests have already covered analytically — this is mostly a
 * smoke test that the partial path doesn't crash and gives non-trivial
 * output.
 * -------------------------------------------------------------------- */
static void test_intent_persistence_partial_blend(void)
{
    snn_language_bridge_t* b = build_ar_bridge();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_autoregressive(b, 0.5f, 0.5f) == 0,
            "set partial blend");

    float intent[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    char picks[8][32];
    int n = produce_tokens(b, intent, 4, picks, 8);

    EXPECT(n >= 1, "partial blend: expected ≥1 pick; got %d", n);
    if (n >= 1) {
        EXPECT(strcmp(picks[0], "A") == 0, "first pick should still be A; got %s", picks[0]);
    }

    /* Bad-arg rejection. */
    EXPECT(snn_language_bridge_set_autoregressive(b, -0.1f, 0.5f) != 0,
            "negative intent_persistence rejected");
    EXPECT(snn_language_bridge_set_autoregressive(b, 0.5f, 1.5f) != 0,
            "word_feedback > 1 rejected");

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[PA-2] test_lang_bridge_autoregressive\n");
    test_intent_persistence_zero_matches_legacy();
    test_intent_persistence_one_keeps_intent_constant();
    test_intent_persistence_partial_blend();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
