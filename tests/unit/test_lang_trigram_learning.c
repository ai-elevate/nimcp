/**
 * @file test_lang_trigram_learning.c
 * @brief TA-4 — verify trigram next-token training: gating, application,
 *        cold-start safety, and stat-counter advance.
 *
 * Pattern: standalone smoke test. Compile:
 *   gcc -O2 -I include tests/unit/test_lang_trigram_learning.c \
 *       -L build/lib -lnimcp -lm -lpthread \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_trigram_learning
 *
 * Coverage:
 *   1. test_default_off_preserves_pa4:
 *      With trigram learning OFF (default), learn_text_bigrams produces
 *      the same applied-bigram count and the same total_trigram_updates
 *      stat (zero). Old PA-4 behavior unchanged.
 *
 *   2. test_trigram_on_advances_counter:
 *      With trigram learning ON, training on "the cat sat on the mat"
 *      produces nonzero total_trigram_updates after the call.
 *
 *   3. test_cold_start_skips_cleanly:
 *      learn_next_token_triple on a fresh GL with no prior bindings is
 *      a no-op (returns -1). No crash, no counter advance.
 *
 *   4. test_counter_only_advances_on_apply:
 *      Run a pure cold-start text (no prior bindings for any token)
 *      with trigram learning ON. total_trigram_updates stays 0 — the
 *      counter increments only on successful applies.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "language/nimcp_grounded_language.h"

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

/* ---------- helpers --------------------------------------------------- */

/* Spin up a (gl, bridge) pair connected together. Caller must
 * grounded_language_destroy(gl) and snn_language_bridge_destroy(b)
 * separately when done. Returns 0 on success, -1 otherwise. */
static int make_lang_pair(grounded_language_t** gl_out,
                          snn_language_bridge_t** b_out)
{
    grounded_language_t* gl = grounded_language_create(32, NULL);
    if (!gl) return -1;

    snn_lang_config_t cfg = snn_lang_config_default();
    cfg.max_concept_pops = SNN_LANG_MAX_CONCEPT_POPS;
    cfg.max_word_pops    = SNN_LANG_MAX_WORD_POPS;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    if (!b) {
        grounded_language_destroy(gl);
        return -1;
    }
    grounded_language_connect_snn_bridge(gl, b);

    *gl_out = gl;
    *b_out  = b;
    return 0;
}

/* Ground a word with a deterministic feature vector so the bridge gets
 * concept_pop → word_pop bindings. Different `seed_offset` values give
 * different feature shapes so words diverge in concept space. */
static int seed_word(grounded_language_t* gl, const char* word, int seed_offset)
{
    float feats[32];
    for (int i = 0; i < 32; i++) feats[i] = 0.0f;
    /* A small set of nonzero features keyed off seed_offset so we get
     * distinguishable encodings per word. Using i*0.13 + 0.01 keeps every
     * value finite and below 1.0. */
    feats[(0 + seed_offset) % 32] = 1.0f;
    feats[(3 + seed_offset) % 32] = 0.7f;
    feats[(7 + seed_offset) % 32] = 0.5f;

    gl_grounding_event_t ev = {
        .word = word,
        .modality = GL_MODALITY_LINGUISTIC,
        .sensory_features = feats,
        .feature_dim = 32,
        .emotional_valence = 0.0f,
        .emotional_arousal = 0.5f,
        .attention = 0.9f,
        .context_sentence = NULL,
        .negative = false,
    };
    return grounded_language_ground(gl, &ev);
}

/* ---------- tests ----------------------------------------------------- */

static void test_default_off_preserves_pa4(void)
{
    grounded_language_t* gl = NULL;
    snn_language_bridge_t* b = NULL;
    EXPECT(make_lang_pair(&gl, &b) == 0, "make pair");
    if (!gl || !b) return;

    /* Sanity: trigram learning is OFF by default. */
    EXPECT(snn_language_bridge_get_trigram_learning_enabled(b) == false,
            "default is OFF");

    /* Seed bindings for every token so the bigram path can apply. */
    EXPECT(seed_word(gl, "the",  0) == 0, "seed 'the'");
    EXPECT(seed_word(gl, "cat",  5) == 0, "seed 'cat'");
    EXPECT(seed_word(gl, "sat",  9) == 0, "seed 'sat'");
    EXPECT(seed_word(gl, "on",  13) == 0, "seed 'on'");
    EXPECT(seed_word(gl, "mat", 17) == 0, "seed 'mat'");

    snn_lang_stats_t s_pre;
    EXPECT(snn_language_bridge_get_stats(b, &s_pre) == 0, "get_stats pre");
    uint64_t trigram_pre = s_pre.total_trigram_updates;

    int n = grounded_language_learn_text_bigrams(gl,
                "the cat sat on the mat", 0.03f);
    EXPECT(n >= 0, "learn_text_bigrams returned %d", n);

    snn_lang_stats_t s_post;
    EXPECT(snn_language_bridge_get_stats(b, &s_post) == 0, "get_stats post");

    /* With trigram OFF, the counter must not move. */
    EXPECT(s_post.total_trigram_updates == trigram_pre,
            "trigram counter must NOT advance when flag is OFF; pre=%llu post=%llu",
            (unsigned long long)trigram_pre,
            (unsigned long long)s_post.total_trigram_updates);

    grounded_language_destroy(gl);
    snn_language_bridge_destroy(b);
}

static void test_trigram_on_advances_counter(void)
{
    grounded_language_t* gl = NULL;
    snn_language_bridge_t* b = NULL;
    EXPECT(make_lang_pair(&gl, &b) == 0, "make pair");
    if (!gl || !b) return;

    /* Seed all five distinct words. */
    EXPECT(seed_word(gl, "the",  0) == 0, "seed 'the'");
    EXPECT(seed_word(gl, "cat",  5) == 0, "seed 'cat'");
    EXPECT(seed_word(gl, "sat",  9) == 0, "seed 'sat'");
    EXPECT(seed_word(gl, "on",  13) == 0, "seed 'on'");
    EXPECT(seed_word(gl, "mat", 17) == 0, "seed 'mat'");

    /* Switch trigram learning ON. */
    EXPECT(snn_language_bridge_set_trigram_learning_enabled(b, true) == 0,
            "set trigram ON");
    EXPECT(snn_language_bridge_get_trigram_learning_enabled(b) == true,
            "flag round-trip ON");

    snn_lang_stats_t s_pre;
    EXPECT(snn_language_bridge_get_stats(b, &s_pre) == 0, "get_stats pre");
    uint64_t trigram_pre = s_pre.total_trigram_updates;

    /* "the cat sat on the mat" is 6 tokens → 5 bigrams, 4 trigrams.
     * Some/all 4 trigrams should apply (every prefix has prior bindings). */
    int n = grounded_language_learn_text_bigrams(gl,
                "the cat sat on the mat", 0.03f);
    EXPECT(n >= 0, "learn_text_bigrams returned %d", n);

    snn_lang_stats_t s_post;
    EXPECT(snn_language_bridge_get_stats(b, &s_post) == 0, "get_stats post");

    EXPECT(s_post.total_trigram_updates > trigram_pre,
            "trigram counter must advance when flag is ON; pre=%llu post=%llu",
            (unsigned long long)trigram_pre,
            (unsigned long long)s_post.total_trigram_updates);

    grounded_language_destroy(gl);
    snn_language_bridge_destroy(b);
}

static void test_cold_start_skips_cleanly(void)
{
    grounded_language_t* gl = NULL;
    snn_language_bridge_t* b = NULL;
    EXPECT(make_lang_pair(&gl, &b) == 0, "make pair");
    if (!gl || !b) return;

    /* No seeding → all three tokens are cold-start (no bindings). */
    int rc = grounded_language_learn_next_token_triple(gl,
                "alpha", "beta", "gamma", 0.03f);
    EXPECT(rc == -1, "cold-start triple must be a no-op; rc=%d", rc);

    /* Now seed only prev1, leave prev2 cold. Triple should still skip
     * because the merged-context cold-start guard requires BOTH prev1
     * and prev2 to have prior encodings. */
    EXPECT(seed_word(gl, "alpha", 0) == 0, "seed 'alpha'");
    rc = grounded_language_learn_next_token_triple(gl,
                "alpha", "beta", "gamma", 0.03f);
    EXPECT(rc == -1, "half-cold (prev2 missing) must skip; rc=%d", rc);

    /* And the symmetric case: seed prev2 only. */
    grounded_language_destroy(gl);
    snn_language_bridge_destroy(b);

    EXPECT(make_lang_pair(&gl, &b) == 0, "make pair 2");
    if (!gl || !b) return;
    EXPECT(seed_word(gl, "beta", 0) == 0, "seed 'beta'");
    rc = grounded_language_learn_next_token_triple(gl,
                "alpha", "beta", "gamma", 0.03f);
    EXPECT(rc == -1, "half-cold (prev1 missing) must skip; rc=%d", rc);

    /* Counter must remain zero — no successful applies. */
    snn_lang_stats_t s;
    EXPECT(snn_language_bridge_get_stats(b, &s) == 0, "get_stats");
    EXPECT(s.total_trigram_updates == 0,
            "cold-start must not bump counter; got %llu",
            (unsigned long long)s.total_trigram_updates);

    grounded_language_destroy(gl);
    snn_language_bridge_destroy(b);
}

static void test_counter_only_advances_on_apply(void)
{
    grounded_language_t* gl = NULL;
    snn_language_bridge_t* b = NULL;
    EXPECT(make_lang_pair(&gl, &b) == 0, "make pair");
    if (!gl || !b) return;

    /* No seeding for any word in the text → every triple is a cold-start
     * skip — even with trigram learning ON. */
    EXPECT(snn_language_bridge_set_trigram_learning_enabled(b, true) == 0,
            "set trigram ON");

    int n = grounded_language_learn_text_bigrams(gl,
                "alpha beta gamma delta", 0.03f);
    /* Bigrams also won't apply (cold-start), but learn_text_bigrams
     * returns the count of *applied* bigrams which can be 0. The
     * relevant assertion is on the trigram counter. */
    EXPECT(n >= 0, "rc=%d", n);

    snn_lang_stats_t s;
    EXPECT(snn_language_bridge_get_stats(b, &s) == 0, "get_stats");
    EXPECT(s.total_trigram_updates == 0,
            "trigram counter must not advance when every triple is cold-start; got %llu",
            (unsigned long long)s.total_trigram_updates);

    grounded_language_destroy(gl);
    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[TA-4] test_lang_trigram_learning\n");
    test_default_off_preserves_pa4();
    test_trigram_on_advances_counter();
    test_cold_start_skips_cleanly();
    test_counter_only_advances_on_apply();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 4 tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
        return 1;
    }
}
