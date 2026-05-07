/**
 * @file test_lang_bridge_sampling.c
 * @brief PA-6 — verify temperature / top_p softmax sampling in bridge_produce.
 *
 * Pattern: standalone smoke test, no GTest dep. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_sampling.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_bridge_sampling
 *
 * Coverage:
 *   1. test_temperature_zero_is_argmax:
 *      4 single-binding words A/B/C/D bound to concepts 0..3. Intent gives
 *      A the strongest cosine score. With temperature=0, A must win the
 *      first-word slot every time across 50 produce calls.
 *
 *   2. test_temperature_high_spreads_distribution:
 *      Same setup, T=10 (very flat softmax). Across 200 first-word picks,
 *      each candidate should appear roughly 25% of the time (each in a
 *      [25, 75] window — generous to avoid flake; mean is ~50).
 *
 *   3. test_top_p_truncation_drops_tail:
 *      Setup with non-uniform scores (concept rates 1.0, 0.5, 0.2, 0.05) and
 *      T=1.0 + top_p=0.55. Cumulative top-2 prob covers the nucleus; D
 *      (rate 0.05) sits firmly in the tail and must NEVER appear in 200
 *      first-word picks.
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

/* Build a bridge with 4 single-binding words A/B/C/D each bound to a
 * distinct concept_pop (0..3) with weight 1.0. After cosine norm, every
 * binding has ‖w‖ = 1, so cosine_score(word_i) = concept_rates[i]. */
static snn_language_bridge_t* build_4words(void)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = 4;
    cfg.max_word_pops    = 4;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    if (!b) return NULL;

    static const char* names[4] = {"A", "B", "C", "D"};
    for (uint32_t i = 0; i < 4; i++) {
        snn_language_bridge_register_concept(b, i, /*concept_id=*/i + 1);
        snn_language_bridge_register_word(b, i, names[i]);
        snn_language_bridge_bind(b, /*concept_pop=*/i, /*word_pop=*/i, 1.0f);
    }
    return b;
}

/* Extract the first whitespace-delimited token from result.text. */
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

static int idx_of(const char* w)
{
    if (!w || !w[0] || w[1]) return -1;
    if (w[0] == 'A') return 0;
    if (w[0] == 'B') return 1;
    if (w[0] == 'C') return 2;
    if (w[0] == 'D') return 3;
    return -1;
}

static void test_temperature_zero_is_argmax(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_sampling(b, 0.0f, 1.0f) == 0,
            "set_sampling T=0");

    /* Concept rates rank A > B > C > D by cosine score. */
    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    int counts[4] = {0};
    const int N = 50;
    for (int trial = 0; trial < N; trial++) {
        snn_lang_production_result_t res;
        memset(&res, 0, sizeof(res));
        int rc = snn_language_bridge_produce(b, intent, 4, &res);
        EXPECT(rc == 0 && res.text, "produce trial %d rc=%d", trial, rc);
        char w[16];
        first_word(res.text, w, sizeof(w));
        int idx = idx_of(w);
        if (idx >= 0) counts[idx]++;
        snn_lang_production_result_cleanup(&res);
    }

    EXPECT(counts[0] == N, "T=0 must always pick A; got A=%d B=%d C=%d D=%d",
            counts[0], counts[1], counts[2], counts[3]);

    snn_language_bridge_destroy(b);
}

static void test_temperature_high_spreads_distribution(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_sampling(b, 10.0f, 1.0f) == 0,
            "set_sampling T=10");

    /* Same intent — but T=10 flattens the softmax to near-uniform over the
     * 4 candidates that actually receive positive concept_rates. */
    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    int counts[4] = {0};
    const int N = 200;
    for (int trial = 0; trial < N; trial++) {
        snn_lang_production_result_t res;
        memset(&res, 0, sizeof(res));
        snn_language_bridge_produce(b, intent, 4, &res);
        char w[16];
        first_word(res.text, w, sizeof(w));
        int idx = idx_of(w);
        if (idx >= 0) counts[idx]++;
        snn_lang_production_result_cleanup(&res);
    }

    fprintf(stderr, "  [T=10] picks: A=%d B=%d C=%d D=%d (of %d)\n",
            counts[0], counts[1], counts[2], counts[3], N);
    /* All four should be reachable. Each has expected ~50 picks (25% of 200);
     * tolerate [10, 90] to absorb sampling noise. */
    for (int i = 0; i < 4; i++) {
        EXPECT(counts[i] >= 10 && counts[i] <= 100,
                "candidate %d should be reachable in [10,100]; got %d",
                i, counts[i]);
    }

    snn_language_bridge_destroy(b);
}

static void test_top_p_truncation_drops_tail(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* T=1.0 + top_p=0.55:
     * scores = [1.0, 0.5, 0.2, 0.05] (cosine).
     * exp = [2.718, 1.648, 1.221, 1.051]; sum=6.638.
     * probs = [0.41, 0.25, 0.18, 0.16].
     * Sorted desc: A=0.41, B=0.25, C=0.18, D=0.16.
     * Cumulative: 0.41 → 0.66 (≥ 0.55, stop) → keep A,B only.
     * D (and C) are zero'd; only A,B sample. */
    EXPECT(snn_language_bridge_set_sampling(b, 1.0f, 0.55f) == 0,
            "set_sampling T=1 top_p=0.55");

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    int counts[4] = {0};
    const int N = 200;
    for (int trial = 0; trial < N; trial++) {
        snn_lang_production_result_t res;
        memset(&res, 0, sizeof(res));
        snn_language_bridge_produce(b, intent, 4, &res);
        char w[16];
        first_word(res.text, w, sizeof(w));
        int idx = idx_of(w);
        if (idx >= 0) counts[idx]++;
        snn_lang_production_result_cleanup(&res);
    }

    fprintf(stderr, "  [top_p=0.55] picks: A=%d B=%d C=%d D=%d (of %d)\n",
            counts[0], counts[1], counts[2], counts[3], N);
    EXPECT(counts[3] == 0, "D must be truncated; got D=%d", counts[3]);
    EXPECT(counts[2] == 0, "C must be truncated; got C=%d", counts[2]);
    EXPECT(counts[0] > 0 && counts[1] > 0,
            "A and B must both be reachable; A=%d B=%d", counts[0], counts[1]);
    EXPECT(counts[0] + counts[1] == N, "A+B should cover all picks; got %d",
            counts[0] + counts[1]);

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[PA-6] test_lang_bridge_sampling\n");
    test_temperature_zero_is_argmax();
    test_temperature_high_spreads_distribution();
    test_top_p_truncation_drops_tail();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
