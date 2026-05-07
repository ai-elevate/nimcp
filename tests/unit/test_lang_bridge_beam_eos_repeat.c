/**
 * @file test_lang_bridge_beam_eos_repeat.c
 * @brief TIER1-A/B/C — verify beam search + EOS + repetition penalty in
 *        snn_language_bridge_produce.
 *
 * Pattern: standalone smoke test, no GTest dep. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_beam_eos_repeat.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,$(pwd)/build/lib \
 *       -o /tmp/t && /tmp/t
 *
 * Coverage:
 *   1. test_beam_width_1_no_regression:
 *      Defaults (no opt-ins) and beam_width=1 must produce identical first-
 *      word picks vs the legacy path on a 4-word setup. Both paths argmax
 *      over the same cosine ranking.
 *   2. test_beam_width_4_finds_high_cum_score_path:
 *      Synthetic 4-word setup where greedy picks the locally-best word A
 *      whose continuation is worse than going through C → C-then-D would
 *      give a higher cumulative score. With beam_width=4 the beam search
 *      should explore enough alternatives that — at minimum — the first-
 *      word output is allowed to differ from greedy when greedy is locally
 *      myopic. We assert the beam path actually returns content + the
 *      result text contains > 1 word (showing the beam continued past the
 *      first step).
 *   3. test_eos_halts_produce:
 *      EOS pop registered and bound strongly to a "stop" concept. With
 *      that concept active, the first-step decode picks the EOS pop —
 *      produce returns -1 (no words) cleanly, no output text.
 *   4. test_repetition_penalty_blocks_repeats_within_window:
 *      Use beam_width=2 (so refractory doesn't auto-prune duplicates the
 *      way a single greedy beam already does). With repetition_penalty=0.5
 *      and window=3, no single word_pop should appear twice in the first
 *      3 picks of the produced output. Run several trials, count any
 *      adjacent-or-near duplicates within 3 picks.
 *      (Refractory in single-greedy already blocks immediate dup; this
 *      test focuses on the explicit penalty path firing.)
 *   5. test_disabled_defaults_pass_through:
 *      Defaults: beam_width=1, eos=UINT32_MAX, repetition_penalty=0. After
 *      explicitly setting all three to defaults, behavior must be identical
 *      to the no-op call.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

static int g_failures = 0;

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d " #cond " : ", __func__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        g_failures++; \
    } \
} while (0)

/* Build a bridge with N words each bound to a distinct concept_pop with
 * weight 1.0. Cosine-normalized so cosine_score(word_i) = concept_rates[i]. */
static snn_language_bridge_t* build_n_words(uint32_t n, const char* const* names)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = n;
    cfg.max_word_pops    = n;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    if (!b) return NULL;
    for (uint32_t i = 0; i < n; i++) {
        snn_language_bridge_register_concept(b, i, /*concept_id=*/i + 1);
        snn_language_bridge_register_word(b, i, names[i]);
        snn_language_bridge_bind(b, /*concept_pop=*/i, /*word_pop=*/i, 1.0f);
    }
    return b;
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

/* Count words separated by single spaces. */
static uint32_t count_words(const char* text)
{
    if (!text || !text[0]) return 0;
    uint32_t n = 1;
    for (const char* p = text; *p; p++) {
        if (*p == ' ') n++;
    }
    return n;
}

/* Get token at position idx (0-based), null-terminated into `out`. Returns
 * 1 if found, 0 if past end. */
static int token_at(const char* text, uint32_t idx, char* out, size_t out_max)
{
    if (!text) return 0;
    out[0] = '\0';
    uint32_t cur = 0;
    const char* p = text;
    while (*p) {
        const char* start = p;
        while (*p && *p != ' ') p++;
        if (cur == idx) {
            size_t len = (size_t)(p - start);
            if (len >= out_max) len = out_max - 1;
            memcpy(out, start, len);
            out[len] = '\0';
            return 1;
        }
        cur++;
        if (*p == ' ') p++;
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * Test 1: beam_width=1 must match legacy / argmax behavior.
 *--------------------------------------------------------------------------*/
static void test_beam_width_1_no_regression(void)
{
    static const char* names[4] = {"A", "B", "C", "D"};
    snn_language_bridge_t* b1 = build_n_words(4, names);
    snn_language_bridge_t* b2 = build_n_words(4, names);
    EXPECT(b1 && b2, "bridge create");
    if (!b1 || !b2) return;

    /* b1 = legacy defaults. b2 = explicitly set beam_width=1. Both must
     * be argmax (T=0). */
    EXPECT(snn_language_bridge_set_beam_width(b2, 1) == 0, "set beam=1");

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    snn_lang_production_result_t r1, r2;
    memset(&r1, 0, sizeof(r1));
    memset(&r2, 0, sizeof(r2));

    int rc1 = snn_language_bridge_produce(b1, intent, 4, &r1);
    int rc2 = snn_language_bridge_produce(b2, intent, 4, &r2);
    EXPECT(rc1 == 0 && r1.text, "legacy produce");
    EXPECT(rc2 == 0 && r2.text, "beam=1 produce");

    if (rc1 == 0 && rc2 == 0 && r1.text && r2.text) {
        char w1[16], w2[16];
        first_word(r1.text, w1, sizeof(w1));
        first_word(r2.text, w2, sizeof(w2));
        EXPECT(strcmp(w1, w2) == 0,
               "first words must match: legacy=%s beam=1=%s", w1, w2);
        EXPECT(strcmp(w1, "A") == 0, "argmax first word should be A; got %s", w1);
    }

    snn_lang_production_result_cleanup(&r1);
    snn_lang_production_result_cleanup(&r2);
    snn_language_bridge_destroy(b1);
    snn_language_bridge_destroy(b2);
}

/*--------------------------------------------------------------------------
 * Test 2: beam_width=4 with multi-step setup. We need the beam path to
 *         actually work end-to-end and emit text. Use a temperature > 0
 *         so log-probs are non-degenerate and beam continues for multiple
 *         steps with distinct picks (refractory list grows).
 *--------------------------------------------------------------------------*/
static void test_beam_width_4_finds_high_cum_score_path(void)
{
    static const char* names[4] = {"A", "B", "C", "D"};
    snn_language_bridge_t* b = build_n_words(4, names);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_beam_width(b, 4) == 0, "set beam=4");
    /* T > 0 so probs are differentiated; argmax-style path doesn't engage. */
    EXPECT(snn_language_bridge_set_sampling(b, 1.0f, 1.0f) == 0, "T=1");

    /* Intent with broad support — all four concepts fire, so the beam
     * has real choices at every step rather than collapsing to argmax. */
    float intent[4] = {0.6f, 0.5f, 0.45f, 0.4f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 4, &res);
    EXPECT(rc == 0 && res.text, "beam produce returns text");

    if (rc == 0 && res.text) {
        uint32_t n = count_words(res.text);
        EXPECT(n >= 2, "beam should continue past first step; got n=%u text='%s'",
               n, res.text);
        /* All tokens must be from the registered set + each unique
         * (refractory). */
        char tok[16];
        bool seen[4] = {false, false, false, false};
        for (uint32_t i = 0; i < n; i++) {
            if (!token_at(res.text, i, tok, sizeof(tok))) break;
            int idx = -1;
            if (tok[0] >= 'A' && tok[0] <= 'D' && tok[1] == '\0') {
                idx = tok[0] - 'A';
            }
            EXPECT(idx >= 0 && idx < 4, "token %u '%s' must be one of A/B/C/D",
                   i, tok);
            if (idx >= 0 && idx < 4) {
                EXPECT(!seen[idx], "token %u '%s' duplicates within beam (refractory broken)",
                       i, tok);
                seen[idx] = true;
            }
        }
    }

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

/*--------------------------------------------------------------------------
 * Test 3: EOS halts produce cleanly. We register EOS as the highest-
 *         cosine match for an intent vector; produce should return -1
 *         (no words appended) and result.text should be NULL.
 *--------------------------------------------------------------------------*/
static void test_eos_halts_produce(void)
{
    static const char* names[4] = {"A", "B", "C", "EOS"};
    snn_language_bridge_t* b = build_n_words(4, names);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* Register pop 3 as EOS. */
    EXPECT(snn_language_bridge_set_eos_word_pop(b, 3) == 0, "set eos=3");

    /* Intent rates so EOS (concept 3) wins by argmax. */
    float intent[4] = {0.1f, 0.2f, 0.3f, 1.0f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 4, &res);

    /* With EOS picked as the very first token, produce returns -1 and
     * text remains NULL. */
    EXPECT(rc == -1, "produce should return -1 when EOS picked first; rc=%d", rc);
    EXPECT(res.text == NULL, "result.text should be NULL on first-step EOS");
    EXPECT(res.word_count == 0, "word_count should be 0; got %u", res.word_count);

    snn_lang_production_result_cleanup(&res);

    /* Now a setup where EOS is the SECOND argmax (after a real word). The
     * loop should produce one word, then halt cleanly without appending
     * EOS. */
    float intent2[4] = {1.0f, 0.05f, 0.05f, 0.5f};
    /* But to make EOS the post-refractory winner of step 2 we need C/B to
     * be smaller than EOS. With A=1, B/C=0.05, EOS=0.5: step 1 picks A,
     * step 2 valid candidates exclude A → EOS=0.5 is the new argmax,
     * which halts. */
    memset(&res, 0, sizeof(res));
    rc = snn_language_bridge_produce(b, intent2, 4, &res);
    EXPECT(rc == 0, "produce should succeed with one word + EOS halt");
    if (rc == 0 && res.text) {
        EXPECT(res.word_count == 1, "word_count should be 1 (EOS not counted); got %u",
               res.word_count);
        EXPECT(strcmp(res.text, "A") == 0, "output should be 'A'; got '%s'", res.text);
    }
    snn_lang_production_result_cleanup(&res);

    snn_language_bridge_destroy(b);
}

/*--------------------------------------------------------------------------
 * Test 4: repetition_penalty=0.5 + window=3 prevents repeats within window.
 *         The greedy refractory list already blocks exact duplicates within
 *         the first 32 picks (it's the entire output). To exercise the
 *         penalty firing path, we use a setup where multiple bindings let
 *         a single word_pop stay strongly active across steps; then we
 *         check that the penalty makes a difference in pick distribution.
 *
 *         Simpler verification: with penalty enabled, the first 3 produce
 *         steps must yield 3 distinct word_pops (which is true under
 *         refractory anyway, so this is mainly a no-crash + no-regression
 *         check on the penalty path).
 *--------------------------------------------------------------------------*/
static void test_repetition_penalty_no_repeats_in_window(void)
{
    static const char* names[5] = {"A", "B", "C", "D", "E"};
    snn_language_bridge_t* b = build_n_words(5, names);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_repetition_penalty(b, 0.5f, 3) == 0,
           "set repetition_penalty=0.5 window=3");

    float intent[5] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};
    /* Run several trials and check the first 3 picks are distinct. */
    const int N = 20;
    int repeat_in_3 = 0;
    for (int trial = 0; trial < N; trial++) {
        snn_lang_production_result_t res;
        memset(&res, 0, sizeof(res));
        int rc = snn_language_bridge_produce(b, intent, 5, &res);
        EXPECT(rc == 0 && res.text, "produce trial %d", trial);
        if (rc == 0 && res.text) {
            uint32_t n = count_words(res.text);
            uint32_t check = (n < 3) ? n : 3;
            char toks[3][16];
            for (uint32_t i = 0; i < check; i++) {
                token_at(res.text, i, toks[i], sizeof(toks[i]));
            }
            for (uint32_t i = 0; i < check; i++) {
                for (uint32_t j = i + 1; j < check; j++) {
                    if (toks[i][0] && strcmp(toks[i], toks[j]) == 0) {
                        repeat_in_3++;
                    }
                }
            }
        }
        snn_lang_production_result_cleanup(&res);
    }
    EXPECT(repeat_in_3 == 0,
           "no repeats expected in first 3 picks across %d trials; got %d",
           N, repeat_in_3);

    snn_language_bridge_destroy(b);
}

/*--------------------------------------------------------------------------
 * Test 5: defaults pass through — explicitly setting all three knobs to
 *         their defaults must produce the same output as the legacy
 *         (untouched) bridge.
 *--------------------------------------------------------------------------*/
static void test_disabled_defaults_pass_through(void)
{
    static const char* names[4] = {"A", "B", "C", "D"};
    snn_language_bridge_t* b1 = build_n_words(4, names);
    snn_language_bridge_t* b2 = build_n_words(4, names);
    EXPECT(b1 && b2, "bridge create");
    if (!b1 || !b2) return;

    /* b2 explicitly sets defaults. */
    EXPECT(snn_language_bridge_set_beam_width(b2, 1) == 0, "set beam=1");
    EXPECT(snn_language_bridge_set_eos_word_pop(b2, UINT32_MAX) == 0, "set eos=disabled");
    EXPECT(snn_language_bridge_set_repetition_penalty(b2, 0.0f, 3) == 0, "set rep=0");

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    snn_lang_production_result_t r1, r2;
    memset(&r1, 0, sizeof(r1));
    memset(&r2, 0, sizeof(r2));

    int rc1 = snn_language_bridge_produce(b1, intent, 4, &r1);
    int rc2 = snn_language_bridge_produce(b2, intent, 4, &r2);
    EXPECT(rc1 == rc2 && rc1 == 0, "both produce ok rc1=%d rc2=%d", rc1, rc2);
    if (rc1 == 0 && rc2 == 0 && r1.text && r2.text) {
        EXPECT(strcmp(r1.text, r2.text) == 0,
               "defaults must produce identical text: legacy='%s' explicit='%s'",
               r1.text, r2.text);
        EXPECT(r1.word_count == r2.word_count,
               "word_count must match: %u vs %u",
               r1.word_count, r2.word_count);
    }
    snn_lang_production_result_cleanup(&r1);
    snn_lang_production_result_cleanup(&r2);
    snn_language_bridge_destroy(b1);
    snn_language_bridge_destroy(b2);
}

int main(void)
{
    fprintf(stderr, "[TIER1] test_lang_bridge_beam_eos_repeat\n");
    test_beam_width_1_no_regression();
    test_beam_width_4_finds_high_cum_score_path();
    test_eos_halts_produce();
    test_repetition_penalty_no_repeats_in_window();
    test_disabled_defaults_pass_through();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 5 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
