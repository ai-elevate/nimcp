/**
 * @file test_lang_bridge_quantum_sampling.c
 * @brief PA-6+ — verify quantum-Monte-Carlo sampling mode in bridge_produce.
 *
 * Pattern: standalone smoke test. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_quantum_sampling.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,$(pwd)/build/lib \
 *       -o /tmp/test_lang_bridge_quantum_sampling
 *
 * Coverage:
 *   1. test_qmc_mode_is_non_deterministic:
 *      Same intent, sampling_mode=2, T=1.0 — across 50 produce calls
 *      we must see at least 2 distinct first-word picks. Argmax (mode 0,
 *      T=0) would always pick the same word.
 *
 *   2. test_qmc_output_in_registered_set:
 *      Across many produce calls under mode 2, every produced word must
 *      be one of the registered word forms (no garbage / OOB pop_id).
 *
 *   3. test_qmc_latency_under_10ms:
 *      Wall-clock 100 produce calls under mode 2 with an 8-word vocab —
 *      average per-call must be < 10 ms on this baseline test fixture.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

static int g_failures = 0;

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d " #cond " : ", __func__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        g_failures++; \
    } \
} while (0)

/* Build a bridge with N single-binding words bound to N distinct concepts. */
static snn_language_bridge_t* build_n_words(uint32_t n)
{
    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = n;
    cfg.max_word_pops    = n;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    if (!b) return NULL;
    static const char* names[8] = {"A", "B", "C", "D", "E", "F", "G", "H"};
    for (uint32_t i = 0; i < n && i < 8; i++) {
        snn_language_bridge_register_concept(b, i, /*concept_id=*/i + 1);
        snn_language_bridge_register_word(b, i, names[i]);
        snn_language_bridge_bind(b, i, i, 1.0f);
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

/* --------------------------------------------------------------------
 * Test 1: mode 2 (q-MC) is non-deterministic over repeated produce calls.
 * Set T=1.0, sampling_mode=2; intent gives non-degenerate distribution
 * over candidates (rates 1.0, 0.5, 0.3, 0.2). With single-shot quantum
 * measurement we expect spread across all 4 candidates over 50 calls.
 * -------------------------------------------------------------------- */
static void test_qmc_mode_is_non_deterministic(void)
{
    snn_language_bridge_t* b = build_n_words(4);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_sampling(b, 1.0f, 1.0f) == 0, "T=1.0");
    EXPECT(snn_language_bridge_set_sampling_mode(b, 2) == 0, "mode=2 (q-MC)");

    /* Non-degenerate intent — every candidate has non-trivial mass. */
    float intent[4] = {1.0f, 0.5f, 0.3f, 0.2f};

    int counts[4] = {0};
    for (int trial = 0; trial < 50; trial++) {
        snn_lang_production_result_t prod = {0};
        int rc = snn_language_bridge_produce(b, intent, 4, &prod);
        EXPECT(rc == 0, "produce trial=%d rc=%d", trial, rc);
        if (rc != 0) continue;
        char w[16];
        first_word(prod.text, w, sizeof(w));
        if (w[0] >= 'A' && w[0] <= 'D') counts[w[0] - 'A']++;
        snn_lang_production_result_cleanup(&prod);
    }

    int distinct = 0;
    for (int i = 0; i < 4; i++) {
        if (counts[i] > 0) distinct++;
    }
    fprintf(stderr, "  [q-MC mode=2] counts: A=%d B=%d C=%d D=%d (distinct=%d)\n",
            counts[0], counts[1], counts[2], counts[3], distinct);
    EXPECT(distinct >= 2,
            "q-MC sampling must be non-deterministic; only %d distinct outputs",
            distinct);

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 2: q-MC output stays in the registered word set.
 * Every produced word string must match one of the 8 registered names.
 * -------------------------------------------------------------------- */
static void test_qmc_output_in_registered_set(void)
{
    snn_language_bridge_t* b = build_n_words(8);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_sampling(b, 1.0f, 1.0f) == 0, "T=1.0");
    EXPECT(snn_language_bridge_set_sampling_mode(b, 2) == 0, "mode=2");

    float intent[8] = {1.0f, 0.7f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.05f};

    static const char* legal[8] = {"A", "B", "C", "D", "E", "F", "G", "H"};
    int oob = 0;
    int produced_any = 0;

    for (int trial = 0; trial < 30; trial++) {
        snn_lang_production_result_t prod = {0};
        int rc = snn_language_bridge_produce(b, intent, 8, &prod);
        if (rc != 0) continue;
        if (!prod.text) {
            snn_lang_production_result_cleanup(&prod);
            continue;
        }
        produced_any++;

        /* Walk every produced word; each must match exactly one legal name. */
        const char* p = prod.text;
        while (*p) {
            while (*p == ' ') p++;
            if (!*p) break;
            char tok[16] = {0};
            int i = 0;
            while (*p && *p != ' ' && i < 15) tok[i++] = *p++;
            tok[i] = '\0';

            bool in_set = false;
            for (int k = 0; k < 8; k++) {
                if (strcmp(tok, legal[k]) == 0) { in_set = true; break; }
            }
            if (!in_set) {
                fprintf(stderr, "  OOB token: %s\n", tok);
                oob++;
            }
        }
        snn_lang_production_result_cleanup(&prod);
    }

    EXPECT(produced_any > 0, "at least one produce should succeed");
    EXPECT(oob == 0,
            "q-MC output must stay in registered word set; got %d OOB", oob);

    snn_language_bridge_destroy(b);
}

/* --------------------------------------------------------------------
 * Test 3: q-MC latency. Average per produce call < 10 ms over 100 calls
 * with an 8-word vocab. Generous bound; the inner loop is O(vocab) so
 * this should be sub-ms on any reasonable machine.
 * -------------------------------------------------------------------- */
static void test_qmc_latency_under_10ms(void)
{
    snn_language_bridge_t* b = build_n_words(8);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_sampling(b, 1.0f, 1.0f) == 0, "T=1.0");
    EXPECT(snn_language_bridge_set_sampling_mode(b, 2) == 0, "mode=2");

    float intent[8] = {1.0f, 0.7f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.05f};

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int n_calls = 100;
    for (int trial = 0; trial < n_calls; trial++) {
        snn_lang_production_result_t prod = {0};
        snn_language_bridge_produce(b, intent, 8, &prod);
        snn_lang_production_result_cleanup(&prod);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                        + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double avg_ms = elapsed_ms / n_calls;
    fprintf(stderr, "  [q-MC latency] %d calls in %.2f ms (avg %.3f ms/call)\n",
            n_calls, elapsed_ms, avg_ms);

    EXPECT(avg_ms < 10.0,
            "q-MC produce avg %.3f ms must be < 10 ms", avg_ms);

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[PA-6+] test_lang_bridge_quantum_sampling\n");
    test_qmc_mode_is_non_deterministic();
    test_qmc_output_in_registered_set();
    test_qmc_latency_under_10ms();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 3 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
