/**
 * @file test_lang_bridge_rng_seed.c
 * @brief Tier-4 #17 — verify snn_language_bridge_set_rng_seed makes
 *        softmax sampling deterministic across runs.
 *
 * Pattern: standalone smoke test, no GTest dep. Compile:
 *   gcc -I include tests/unit/test_lang_bridge_rng_seed.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_bridge_rng_seed
 *
 * Coverage:
 *   1. determinism: seed=42 + identical prompt + T>0 produces the SAME
 *      sequence of first-word picks across two bridges.
 *   2. distinctness: a different seed gives a different sequence (NOT just
 *      argmax — confirms sampling is actually consuming the RNG).
 *   3. seed=0 remap: setting seed=0 must succeed (no error) and produce a
 *      non-degenerate sequence (xorshift64 must not get stuck at 0).
 *   4. NULL safety: set_rng_seed(NULL, 42) returns -1.
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

/* Same setup as test_lang_bridge_sampling: 4 single-binding words A/B/C/D
 * each bound to a distinct concept_pop with weight 1.0. */
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
    /* Sampling at modest temperature so picks actually consume the RNG. */
    snn_language_bridge_set_sampling(b, 1.0f, 1.0f);
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

/* Run N produce calls, capturing first-word picks into `picks` (caller
 * supplies a buffer of length >= N). */
static void capture_picks(snn_language_bridge_t* b, int N, char picks[][8])
{
    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    for (int i = 0; i < N; i++) {
        snn_lang_production_result_t res;
        memset(&res, 0, sizeof(res));
        int rc = snn_language_bridge_produce(b, intent, 4, &res);
        EXPECT(rc == 0, "produce trial %d rc=%d", i, rc);
        first_word(res.text, picks[i], 8);
        snn_lang_production_result_cleanup(&res);
    }
}

static void test_determinism_same_seed(void)
{
    const int N = 30;
    char picks_a[N][8];
    char picks_b[N][8];

    snn_language_bridge_t* a = build_4words();
    snn_language_bridge_t* b = build_4words();
    EXPECT(a != NULL && b != NULL, "bridges create");
    if (!a || !b) return;

    EXPECT(snn_language_bridge_set_rng_seed(a, 42ULL) == 0, "seed a");
    EXPECT(snn_language_bridge_set_rng_seed(b, 42ULL) == 0, "seed b");

    capture_picks(a, N, picks_a);
    capture_picks(b, N, picks_b);

    int matches = 0;
    for (int i = 0; i < N; i++) {
        if (strcmp(picks_a[i], picks_b[i]) == 0) matches++;
    }
    fprintf(stderr, "  [seed=42] %d/%d matches\n", matches, N);
    EXPECT(matches == N,
            "all %d picks must match between identically-seeded bridges; got %d",
            N, matches);

    snn_language_bridge_destroy(a);
    snn_language_bridge_destroy(b);
}

static void test_different_seeds_diverge(void)
{
    const int N = 30;
    char picks_a[N][8];
    char picks_b[N][8];

    snn_language_bridge_t* a = build_4words();
    snn_language_bridge_t* b = build_4words();
    EXPECT(a != NULL && b != NULL, "bridges create");
    if (!a || !b) return;

    EXPECT(snn_language_bridge_set_rng_seed(a, 42ULL) == 0, "seed a");
    EXPECT(snn_language_bridge_set_rng_seed(b, 1729ULL) == 0, "seed b");

    capture_picks(a, N, picks_a);
    capture_picks(b, N, picks_b);

    int diffs = 0;
    for (int i = 0; i < N; i++) {
        if (strcmp(picks_a[i], picks_b[i]) != 0) diffs++;
    }
    fprintf(stderr, "  [seed=42 vs 1729] %d/%d differing picks\n", diffs, N);
    /* T=1.0 is a moderate flatten; with random prefix divergence we expect
     * a non-trivial fraction of mismatches. Tolerate ≥ 3 (very generous). */
    EXPECT(diffs >= 3,
            "different seeds must produce distinguishable sequences; got %d",
            diffs);

    snn_language_bridge_destroy(a);
    snn_language_bridge_destroy(b);
}

static void test_seed_zero_remap(void)
{
    /* seed=0 is the xorshift64 fixed-zero attractor; the API must remap. */
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_rng_seed(b, 0ULL) == 0,
            "seed=0 must succeed (remap to 1)");

    const int N = 20;
    char picks[N][8];
    capture_picks(b, N, picks);

    /* If the RNG was stuck at 0, every pick collapses to the same word
     * (typically the argmax). Verify we get at least 2 distinct picks
     * across N tries — proves the remap took effect. */
    int distinct = 0;
    for (int i = 0; i < N; i++) {
        bool seen = false;
        for (int j = 0; j < i; j++) {
            if (strcmp(picks[i], picks[j]) == 0) { seen = true; break; }
        }
        if (!seen) distinct++;
    }
    fprintf(stderr, "  [seed=0 remap] %d distinct picks of %d trials\n",
            distinct, N);
    EXPECT(distinct >= 2,
            "seed=0 remap must yield non-degenerate sampling; got %d distinct",
            distinct);

    snn_language_bridge_destroy(b);
}

static void test_null_safety(void)
{
    EXPECT(snn_language_bridge_set_rng_seed(NULL, 42ULL) == -1,
            "NULL bridge must return -1");
}

int main(void)
{
    fprintf(stderr, "[Tier-4 #17] test_lang_bridge_rng_seed\n");
    test_determinism_same_seed();
    test_different_seeds_diverge();
    test_seed_zero_remap();
    test_null_safety();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 4 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
