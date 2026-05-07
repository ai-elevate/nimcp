/**
 * @file test_lang_bridge_quantum_shannon.c
 * @brief DK-A+ — verify quantum-Shannon entropy_confidence on production result.
 *
 * Pattern: standalone smoke test. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_quantum_shannon.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,$(pwd)/build/lib \
 *       -o /tmp/test_lang_bridge_quantum_shannon
 *
 * Coverage:
 *   1. test_peaked_posterior_high_entropy_confidence:
 *      Geometric-decay intent (rates 1.0, 0.5, 0.25, 0.125), T=0.1,
 *      autoregressive intent_persistence=1 (so refractory steps keep
 *      the same peaked input minus the winner). The post-softmax
 *      posterior is sharply peaked at every step → entropy_confidence
 *      averaged across steps should be in (0.5, 1.0]; we assert ≥ 0.5.
 *
 *   2. test_flat_posterior_low_entropy_confidence:
 *      Intent equally distributed (rates 1.0, 1.0, 1.0, 1.0). Under
 *      softmax T=1.0 the posterior stays uniform → entropy_confidence
 *      should be ≤ 0.1.
 *
 *   3. test_peaked_higher_than_flat:
 *      Stricter relative comparison — for the same fixture, peaked
 *      entropy_confidence must be strictly greater than flat by a
 *      meaningful margin (≥ 0.3).
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
        snn_language_bridge_bind(b, i, i, 1.0f);
    }
    return b;
}

/* --------------------------------------------------------------------
 * Test 1: peaked posterior → high entropy_confidence.
 * Intent gives candidate 0 a heavy weight; T=0.1 sharpens the softmax
 * further. The post-softmax distribution should be very peaked, so
 * entropy_confidence (= 1 − H/log2(K)) should be near 1.
 * -------------------------------------------------------------------- */
static void test_peaked_posterior_high_entropy_confidence(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* Low T → sharp softmax. Mode 1 forces softmax (so we always
     * compute the posterior). */
    EXPECT(snn_language_bridge_set_sampling(b, 0.1f, 1.0f) == 0, "T=0.1");
    EXPECT(snn_language_bridge_set_sampling_mode(b, 1) == 0, "mode=1");
    /* Pin intent across the produce loop so refractory step k still sees
     * the original peaked distribution, just with the winner removed.
     * Without this PA-2 evolves concept_acts and the second/third step's
     * posteriors flatten as state drifts. */
    EXPECT(snn_language_bridge_set_autoregressive(b, 1.0f, 0.0f) == 0,
            "intent_persistence=1, word_feedback=0");

    /* Geometric decay across candidates → produce loop sees a peaked
     * distribution at every step (after refractory removes the winner,
     * the next-best is still 2x the others — under T=0.1 that's a
     * heavily-peaked softmax too). */
    float intent[4] = {1.0f, 0.5f, 0.25f, 0.125f};

    /* Average over a few trials to reduce sampling variance — the
     * entropy_confidence is itself an EMA over the produced steps so
     * a single produce gives one number, but produce can pick multiple
     * words and average. */
    float sum_ec = 0.0f;
    int n_trials = 5;
    for (int i = 0; i < n_trials; i++) {
        snn_lang_production_result_t prod = {0};
        int rc = snn_language_bridge_produce(b, intent, 4, &prod);
        EXPECT(rc == 0, "produce trial=%d rc=%d", i, rc);
        if (rc == 0) {
            sum_ec += prod.entropy_confidence;
        }
        snn_lang_production_result_cleanup(&prod);
    }
    float avg_ec = sum_ec / (float)n_trials;
    fprintf(stderr, "  [peaked] avg entropy_confidence = %.4f\n", avg_ec);
    EXPECT(avg_ec >= 0.5f,
            "peaked posterior should have ec ≥ 0.5; got %.4f", avg_ec);

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 3: peaked entropy_confidence is meaningfully larger than flat.
 * Same softmax-1 mode, same 4-word vocab. Only the intent shape changes.
 * Peaked - flat must be ≥ 0.3 (much larger than any noise floor).
 * -------------------------------------------------------------------- */
static void test_peaked_higher_than_flat(void)
{
    snn_language_bridge_t* b1 = build_4words();
    snn_language_bridge_t* b2 = build_4words();
    EXPECT(b1 && b2, "bridge create");
    if (!b1 || !b2) return;

    snn_language_bridge_set_sampling(b1, 0.1f, 1.0f);
    snn_language_bridge_set_sampling_mode(b1, 1);
    snn_language_bridge_set_autoregressive(b1, 1.0f, 0.0f);

    snn_language_bridge_set_sampling(b2, 1.0f, 1.0f);
    snn_language_bridge_set_sampling_mode(b2, 1);
    snn_language_bridge_set_autoregressive(b2, 1.0f, 0.0f);

    float peaked[4] = {1.0f, 0.5f, 0.25f, 0.125f};
    float flat[4]   = {1.0f, 1.0f, 1.0f, 1.0f};

    /* One produce each, average over a few. */
    float p_sum = 0.0f, f_sum = 0.0f;
    int n = 5;
    for (int i = 0; i < n; i++) {
        snn_lang_production_result_t pp = {0}, ff = {0};
        snn_language_bridge_produce(b1, peaked, 4, &pp);
        snn_language_bridge_produce(b2, flat,   4, &ff);
        p_sum += pp.entropy_confidence;
        f_sum += ff.entropy_confidence;
        snn_lang_production_result_cleanup(&pp);
        snn_lang_production_result_cleanup(&ff);
    }
    float p_avg = p_sum / n;
    float f_avg = f_sum / n;
    fprintf(stderr, "  [peaked vs flat] %.4f vs %.4f (Δ=%.4f)\n",
            p_avg, f_avg, p_avg - f_avg);
    EXPECT((p_avg - f_avg) >= 0.3f,
            "peaked - flat must be ≥ 0.3; got Δ=%.4f", p_avg - f_avg);

    snn_language_bridge_destroy(b1);
    snn_language_bridge_destroy(b2);
}

/* --------------------------------------------------------------------
 * Test 2: flat posterior → low entropy_confidence.
 * Intent equally distributed across all candidates. Under softmax T=1.0
 * the distribution stays close to uniform, so entropy ≈ log2(K) and
 * entropy_confidence ≈ 0.
 * -------------------------------------------------------------------- */
static void test_flat_posterior_low_entropy_confidence(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_sampling(b, 1.0f, 1.0f) == 0, "T=1.0");
    EXPECT(snn_language_bridge_set_sampling_mode(b, 1) == 0, "mode=1");

    /* Uniform intent — every candidate gets the same cosine score,
     * softmax stays flat. */
    float intent[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    float sum_ec = 0.0f;
    int n_trials = 5;
    for (int i = 0; i < n_trials; i++) {
        snn_lang_production_result_t prod = {0};
        int rc = snn_language_bridge_produce(b, intent, 4, &prod);
        EXPECT(rc == 0, "produce trial=%d rc=%d", i, rc);
        if (rc == 0) {
            sum_ec += prod.entropy_confidence;
        }
        snn_lang_production_result_cleanup(&prod);
    }
    float avg_ec = sum_ec / (float)n_trials;
    fprintf(stderr, "  [flat] avg entropy_confidence = %.4f\n", avg_ec);
    EXPECT(avg_ec <= 0.1f,
            "flat posterior should have low entropy_confidence; got %.4f",
            avg_ec);

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[DK-A+] test_lang_bridge_quantum_shannon\n");
    test_peaked_posterior_high_entropy_confidence();
    test_flat_posterior_low_entropy_confidence();
    test_peaked_higher_than_flat();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
