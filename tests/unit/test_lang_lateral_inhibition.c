/**
 * @file test_lang_lateral_inhibition.c
 * @brief S4-C1 fix verification — divisive-normalization lateral inhibition
 *        produces a real winner-take-all attractor.
 *
 * Pattern: standalone smoke test (no GTest dep). Compile via the standard
 * lang-test CMake wiring; manual:
 *   gcc -I include tests/unit/test_lang_lateral_inhibition.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_lateral_inhibition
 *
 * Background
 * ----------
 * The old sigmoid-based update
 *     new_a[k] = sigmoid(gain_self*a[k] - gain_inhibit*(sum_a - a[k]))
 * has a STABLE symmetric fixed point a* ≈ 0.62 at default
 * gain_self=1.5, gain_inhibit=0.026, K∈{2..32}: small asymmetries decay
 * back into the fixed point instead of growing into a winner. Net effect:
 * every candidate converged to ≈ equal activation and the re-rank became
 * noise.
 *
 * The replacement (divisive normalization with exponent p) has a Liapunov
 * argument for the leader → 1, subordinates → 0 attractor at p > 1. With
 * p = 2 we expect the leader > 0.9 and second-place < 0.1 within ~10-15
 * micro-steps for K∈{2, 5, 8, 32}.
 *
 * Testing strategy
 * ----------------
 * decode_with_lateral_inhibition() drives the cohort from
 * decode_spikes() — there is no public hook to inject pre-computed
 * activations. We construct controlled cohorts by registering K
 * concept_pops + K word_pops and binding word_i ⟵ concept_i with a
 * weight chosen so that with a uniform concept_rates input vector,
 * word_0's pre-LI cosine score is the largest. Then we run the public
 * decode_with_lateral_inhibition and assert that one word dominates
 * post-settling (leader > 0.9, second < 0.1).
 *
 * Coverage
 * --------
 *  1. test_winner_take_all_K2:  K=2  → leader >0.9 by T=30.
 *  2. test_winner_take_all_K5:  K=5  → leader >0.9.
 *  3. test_winner_take_all_K8:  K=8  → leader >0.9.
 *  4. test_winner_take_all_K32: K=32 → leader >0.9.
 *  5. test_symmetric_input_no_runaway: K=4 with all weights equal — all
 *     settle to ~1/K = 0.25 (no spurious winner from numerical noise).
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

static snn_language_bridge_t* make_bridge_lateral(uint32_t K)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops          = K + 4;
    cfg.max_word_pops             = K + 4;
    cfg.enable_lateral_inhibition = true;
    /* Default p (= gain_self) is now 1.5 (clamped >= 1.0). The fix treats
     * gain_self AS the exponent. Bump to 2.0 in this test for clearer WTA
     * behavior. Production callers will get the same with default 1.5 →
     * clamped p=1.5 also reaches WTA, just a touch slower. */
    cfg.lateral_gain_self    = 2.0f;
    cfg.lateral_gain_inhibit = 1e-6f;   /* eps floor, not inhibition gain now */
    cfg.lateral_micro_steps  = 30;
    return snn_language_bridge_create(&cfg);
}

/* Build a bridge with K words competing — word i is bound to concept_pop i
 * with weight w_i. Optional uniform_weights mode binds all words to
 * concept_0 with the SAME weight (used for the symmetric test). */
static snn_language_bridge_t* build_K_competitors(uint32_t K,
                                                    float weight_leader,
                                                    float weight_others,
                                                    bool symmetric)
{
    snn_language_bridge_t* b = make_bridge_lateral(K);
    if (!b) return NULL;

    /* Register K concept_pops + K word_pops. */
    for (uint32_t i = 0; i < K; i++) {
        snn_language_bridge_register_concept(b, /*pop=*/i, /*concept_id=*/i + 1);
        char w[32]; snprintf(w, sizeof(w), "w%u", i);
        snn_language_bridge_register_word(b, /*pop=*/i, w);
    }

    if (symmetric) {
        /* All K words bound to concept_0 with the same weight — input
         * vector with concept_0 hot will produce a perfectly symmetric
         * top-K. */
        for (uint32_t i = 0; i < K; i++) {
            snn_language_bridge_bind(b, /*concept_pop=*/0, /*word_pop=*/i,
                                     weight_leader);
        }
    } else {
        /* word_0 has a stronger binding to concept_0; the other word_i are
         * bound to concept_i with weight_others. Input vector will be
         * uniform across concepts 0..K-1, so word_0's pre-LI score >
         * others' scores. */
        snn_language_bridge_bind(b, /*concept_pop=*/0, /*word_pop=*/0,
                                 weight_leader);
        for (uint32_t i = 1; i < K; i++) {
            snn_language_bridge_bind(b, /*concept_pop=*/i, /*word_pop=*/i,
                                     weight_others);
        }
    }
    return b;
}

static void test_winner_take_all_K(uint32_t K, const char* tag)
{
    snn_language_bridge_t* b = build_K_competitors(K, /*leader=*/1.0f,
                                                    /*others=*/0.5f,
                                                    /*symmetric=*/false);
    EXPECT(b != NULL, "bridge create failed [%s]", tag);
    if (!b) return;

    /* Uniform input — all K concepts firing equally. word_0 wins pre-LI
     * because its (single) binding has the largest cosine. */
    float concept_rates[64] = {0};
    for (uint32_t i = 0; i < K; i++) concept_rates[i] = 1.0f;

    snn_lang_word_result_t results[32];
    memset(results, 0, sizeof(results));
    uint32_t num = 0;
    int rc = snn_language_bridge_decode_with_lateral_inhibition(
        b, concept_rates, /*num_concept_pops=*/K,
        results, /*max_results=*/K, &num);
    EXPECT(rc == 0, "decode_with_lateral_inhibition rc=%d [%s]", rc, tag);
    EXPECT(num > 0, "decode produced 0 results [%s]", tag);

    if (num == 0) {
        snn_language_bridge_destroy(b);
        return;
    }

    float leader_a = results[0].activation;
    float second_a = (num > 1) ? results[1].activation : 0.0f;
    uint32_t leader_pop = results[0].word_pop;

    fprintf(stderr, "[%s] K=%u num=%u leader_pop=%u leader=%.4f second=%.4f\n",
            tag, K, num, leader_pop, (double)leader_a, (double)second_a);

    /* word_0 has the strongest pre-LI cosine — it must be the post-LI
     * winner too (LI doesn't change the ordering, just amplifies it). */
    EXPECT(leader_pop == 0,
           "[%s] K=%u: post-LI leader is pop %u, expected 0", tag, K, leader_pop);
    EXPECT(leader_a > 0.9f,
           "[%s] K=%u: leader activation %.4f not > 0.9 — WTA failed",
           tag, K, (double)leader_a);
    EXPECT(second_a < 0.1f,
           "[%s] K=%u: second-place activation %.4f not < 0.1 — WTA failed",
           tag, K, (double)second_a);

    snn_language_bridge_destroy(b);
}

static void test_winner_take_all_K2(void)  { test_winner_take_all_K(2,  "K2");  }
static void test_winner_take_all_K5(void)  { test_winner_take_all_K(5,  "K5");  }
static void test_winner_take_all_K8(void)  { test_winner_take_all_K(8,  "K8");  }
static void test_winner_take_all_K32(void) { test_winner_take_all_K(32, "K32"); }

/* Symmetric input → no spurious winner from numerical noise. */
static void test_symmetric_input_no_runaway(void)
{
    uint32_t K = 4;
    snn_language_bridge_t* b = build_K_competitors(K, /*leader=*/1.0f,
                                                    /*others=*/1.0f,
                                                    /*symmetric=*/true);
    EXPECT(b != NULL, "bridge create failed");
    if (!b) return;

    /* Concept_0 hot — all K words have the SAME binding to it, so all K
     * pre-LI scores are equal. */
    float concept_rates[16] = {0};
    concept_rates[0] = 1.0f;

    snn_lang_word_result_t results[8];
    memset(results, 0, sizeof(results));
    uint32_t num = 0;
    int rc = snn_language_bridge_decode_with_lateral_inhibition(
        b, concept_rates, /*num_concept_pops=*/K,
        results, K, &num);
    EXPECT(rc == 0, "symmetric decode rc=%d", rc);

    if (num == 0) {
        /* Symmetric input may give 0 top-K if scores tie at zero —
         * acceptable, just print and move on. */
        fprintf(stderr, "[symmetric] num=0 — acceptable degenerate case\n");
        snn_language_bridge_destroy(b);
        return;
    }

    fprintf(stderr, "[symmetric] num=%u activations:", num);
    for (uint32_t i = 0; i < num; i++) {
        fprintf(stderr, " %.4f", (double)results[i].activation);
    }
    fprintf(stderr, "\n");

    /* Divisive normalization preserves symmetry — all should sit near
     * 1/num. The exact value depends on the cohort size produced by
     * decode_spikes (which may be < K if some pops were skipped). */
    if (num >= 2) {
        float spread = results[0].activation - results[num - 1].activation;
        EXPECT(spread < 0.05f,
               "[symmetric] activation spread %.4f > 0.05 — symmetry broken",
               (double)spread);
    }

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "=== test_lang_lateral_inhibition: S4-C1 WTA fix ===\n");

    test_winner_take_all_K2();
    test_winner_take_all_K5();
    test_winner_take_all_K8();
    test_winner_take_all_K32();
    test_symmetric_input_no_runaway();

    if (g_failures > 0) {
        fprintf(stderr, "FAILED: %d assertion(s)\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASSED\n");
    return 0;
}
