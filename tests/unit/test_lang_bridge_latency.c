/**
 * @file test_lang_bridge_latency.c
 * @brief Tier-4 #16 — verify bridge_produce records produce_total_us /
 *        produce_call_count and that get_avg_produce_us is finite.
 *
 * Pattern: standalone smoke test, no GTest dep. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_latency.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_bridge_latency
 *
 * Coverage:
 *   1. baseline: fresh bridge → produce_call_count == 0,
 *      produce_total_us == 0, avg_produce_us == 0.
 *   2. produce 10 times → produce_call_count == 10,
 *      produce_total_us > 0, avg_produce_us == total/count and is finite.
 *   3. reset_stats zeroes the latency counters.
 *   4. NULL/invalid bridge → get_avg_produce_us returns 0.0f.
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

    static const char* names[4] = {"alpha", "beta", "gamma", "delta"};
    for (uint32_t i = 0; i < 4; i++) {
        snn_language_bridge_register_concept(b, i, /*concept_id=*/i + 1);
        snn_language_bridge_register_word(b, i, names[i]);
        snn_language_bridge_bind(b, i, i, 1.0f);
    }
    return b;
}

static void test_baseline_zero(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    snn_lang_stats_t s;
    EXPECT(snn_language_bridge_get_stats(b, &s) == 0, "get_stats");
    EXPECT(s.produce_call_count == 0,
            "fresh bridge produce_call_count must be 0; got %llu",
            (unsigned long long)s.produce_call_count);
    EXPECT(s.produce_total_us == 0,
            "fresh bridge produce_total_us must be 0; got %llu",
            (unsigned long long)s.produce_total_us);

    float avg = snn_language_bridge_get_avg_produce_us(b);
    EXPECT(avg == 0.0f, "fresh bridge avg_produce_us must be 0; got %.6f", avg);

    snn_language_bridge_destroy(b);
}

static void test_n_produce_calls(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};

    const int N = 10;
    for (int i = 0; i < N; i++) {
        snn_lang_production_result_t res;
        memset(&res, 0, sizeof(res));
        int rc = snn_language_bridge_produce(b, intent, 4, &res);
        EXPECT(rc == 0, "produce trial %d rc=%d", i, rc);
        snn_lang_production_result_cleanup(&res);
    }

    snn_lang_stats_t s;
    EXPECT(snn_language_bridge_get_stats(b, &s) == 0, "get_stats");
    EXPECT(s.produce_call_count == (uint64_t)N,
            "produce_call_count must equal %d; got %llu",
            N, (unsigned long long)s.produce_call_count);
    EXPECT(s.produce_total_us > 0,
            "produce_total_us must be > 0; got %llu",
            (unsigned long long)s.produce_total_us);

    float avg = snn_language_bridge_get_avg_produce_us(b);
    /* avg should equal total/count and be finite + positive. */
    float expected = (float)s.produce_total_us / (float)s.produce_call_count;
    EXPECT(isfinite(avg), "avg_produce_us must be finite; got %.6f", avg);
    EXPECT(avg > 0.0f, "avg_produce_us must be > 0 with timed calls; got %.6f", avg);
    EXPECT(fabsf(avg - expected) < 1e-3f,
            "avg should match total/count: avg=%.6f expected=%.6f",
            avg, expected);
    fprintf(stderr, "  [latency] %d calls, total=%llu us, avg=%.3f us/call\n",
            N, (unsigned long long)s.produce_total_us, avg);

    /* test 3: reset_stats zeroes latency counters. */
    EXPECT(snn_language_bridge_reset_stats(b) == 0, "reset_stats");
    snn_lang_stats_t s2;
    EXPECT(snn_language_bridge_get_stats(b, &s2) == 0, "get_stats post-reset");
    EXPECT(s2.produce_call_count == 0,
            "post-reset produce_call_count must be 0; got %llu",
            (unsigned long long)s2.produce_call_count);
    EXPECT(s2.produce_total_us == 0,
            "post-reset produce_total_us must be 0; got %llu",
            (unsigned long long)s2.produce_total_us);
    EXPECT(snn_language_bridge_get_avg_produce_us(b) == 0.0f,
            "post-reset avg_produce_us must be 0");

    snn_language_bridge_destroy(b);
}

static void test_null_safety(void)
{
    EXPECT(snn_language_bridge_get_avg_produce_us(NULL) == 0.0f,
            "NULL bridge avg must be 0.0");
}

int main(void)
{
    fprintf(stderr, "[Tier-4 #16] test_lang_bridge_latency\n");
    test_baseline_zero();
    test_n_produce_calls();
    test_null_safety();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
