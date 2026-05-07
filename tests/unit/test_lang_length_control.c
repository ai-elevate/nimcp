/**
 * @file test_lang_length_control.c
 * @brief TB-7 — verify hard length-control on snn_language_bridge_produce.
 *
 * Compile (ad-hoc):
 *   gcc -I include tests/unit/test_lang_length_control.c \
 *       -L build/lib -lnimcp -lm \
 *       -Wl,-rpath,$(pwd)/build/lib \
 *       -o /tmp/t && /tmp/t
 *
 * Coverage:
 *   1. test_default_off_no_regression:
 *      Both knobs at sentinel 0 — produce output is bit-identical between
 *      a fresh-default bridge and a bridge whose knobs were explicitly
 *      reset to 0,0. Stats counters length_min_suppressions and
 *      length_max_truncations stay at 0.
 *
 *   2. test_max_words_caps_output:
 *      With max_produce_words = 3 on a 10-word setup, produce returns at
 *      most 3 words and length_max_truncations bumps once.
 *
 *   3. test_min_words_suppresses_eos:
 *      Register an EOS pop that would otherwise fire on the first decode.
 *      With min_produce_words = 5, the EOS is suppressed at least once,
 *      length_min_suppressions advances, and the produced word_count is
 *      at least 5.
 *
 *   4. test_setter_validation:
 *      - NULL bridge → -1.
 *      - min > max with both nonzero → -1, config unchanged.
 *      - min ≤ max accepted; either side at 0 (disabled) accepted with
 *        the other side nonzero.
 *      - Values above the 1024 clamp are silently capped.
 *      - Getter copies both values; NULL out-pointers tolerated.
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
 * weight 1.0. Cosine-normalized so cosine_score(word_i) ≈ concept_rates[i]. */
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

static uint32_t count_words(const char* text)
{
    if (!text || !text[0]) return 0;
    uint32_t n = 1;
    for (const char* p = text; *p; p++) {
        if (*p == ' ') n++;
    }
    return n;
}

/*--------------------------------------------------------------------------
 * Test 1: defaults pass-through. Bit-identical produce vs explicit zeros.
 *--------------------------------------------------------------------------*/
static void test_default_off_no_regression(void)
{
    static const char* names[5] = {"A", "B", "C", "D", "E"};
    snn_language_bridge_t* b1 = build_n_words(5, names);
    snn_language_bridge_t* b2 = build_n_words(5, names);
    EXPECT(b1 && b2, "bridge create");
    if (!b1 || !b2) return;

    /* b2 explicitly resets to defaults. */
    EXPECT(snn_language_bridge_set_length_control(b2, 0, 0) == 0,
           "set length_control(0,0)");

    /* Both bridges seeded with the same RNG so any sampling-mode test
     * stays deterministic — we use argmax (T=0 default) here so seed
     * doesn't strictly matter, but better to be explicit. */
    snn_language_bridge_set_rng_seed(b1, 42);
    snn_language_bridge_set_rng_seed(b2, 42);

    float intent[5] = {1.0f, 0.5f, 0.2f, 0.1f, 0.05f};
    snn_lang_production_result_t r1, r2;
    memset(&r1, 0, sizeof(r1));
    memset(&r2, 0, sizeof(r2));

    int rc1 = snn_language_bridge_produce(b1, intent, 5, &r1);
    int rc2 = snn_language_bridge_produce(b2, intent, 5, &r2);
    EXPECT(rc1 == 0 && r1.text, "legacy produce rc=%d text=%p",
           rc1, (void*)r1.text);
    EXPECT(rc2 == 0 && r2.text, "explicit-defaults produce rc=%d text=%p",
           rc2, (void*)r2.text);

    if (rc1 == 0 && rc2 == 0 && r1.text && r2.text) {
        EXPECT(strcmp(r1.text, r2.text) == 0,
               "default vs explicit-zero text must match: '%s' vs '%s'",
               r1.text, r2.text);
        EXPECT(r1.word_count == r2.word_count,
               "word_count mismatch: %u vs %u",
               r1.word_count, r2.word_count);
    }

    /* Stats counters must stay at 0 — no suppression / truncation
     * occurred since both knobs were OFF. */
    snn_lang_stats_t s1, s2;
    snn_language_bridge_get_stats(b1, &s1);
    snn_language_bridge_get_stats(b2, &s2);
    EXPECT(s1.length_min_suppressions == 0,
           "b1 length_min_suppressions=%llu",
           (unsigned long long)s1.length_min_suppressions);
    EXPECT(s1.length_max_truncations == 0,
           "b1 length_max_truncations=%llu",
           (unsigned long long)s1.length_max_truncations);
    EXPECT(s2.length_min_suppressions == 0,
           "b2 length_min_suppressions=%llu",
           (unsigned long long)s2.length_min_suppressions);
    EXPECT(s2.length_max_truncations == 0,
           "b2 length_max_truncations=%llu",
           (unsigned long long)s2.length_max_truncations);

    snn_lang_production_result_cleanup(&r1);
    snn_lang_production_result_cleanup(&r2);
    snn_language_bridge_destroy(b1);
    snn_language_bridge_destroy(b2);
}

/*--------------------------------------------------------------------------
 * Test 2: max_produce_words = 3 → at most 3 words emitted.
 *--------------------------------------------------------------------------*/
static void test_max_words_caps_output(void)
{
    static const char* names[10] = {"a", "b", "c", "d", "e",
                                     "f", "g", "h", "i", "j"};
    snn_language_bridge_t* b = build_n_words(10, names);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_length_control(b, 0, 3) == 0,
           "set max=3");

    /* Verify the getter reads it back. */
    uint32_t got_min = 99, got_max = 99;
    EXPECT(snn_language_bridge_get_length_control(b, &got_min, &got_max) == 0,
           "get length_control");
    EXPECT(got_min == 0 && got_max == 3,
           "getter min=%u max=%u", got_min, got_max);

    /* Broad intent so all 10 words are viable candidates — the loop would
     * naturally run for many steps without the cap. */
    float intent[10] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f,
                        0.75f, 0.7f, 0.65f, 0.6f, 0.55f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 10, &res);
    EXPECT(rc == 0 && res.text, "produce rc=%d", rc);

    if (rc == 0 && res.text) {
        uint32_t n = count_words(res.text);
        EXPECT(n <= 3, "max_words cap should be ≤3 words; got %u text='%s'",
               n, res.text);
        EXPECT(res.word_count <= 3,
               "result.word_count should be ≤3; got %u", res.word_count);
    }

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.length_max_truncations == 1,
           "length_max_truncations should bump exactly once; got %llu",
           (unsigned long long)s.length_max_truncations);
    EXPECT(s.length_min_suppressions == 0,
           "length_min_suppressions should stay 0; got %llu",
           (unsigned long long)s.length_min_suppressions);

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

/*--------------------------------------------------------------------------
 * Test 3: min_produce_words suppresses an early EOS.
 *
 * Setup: 6 words, the highest-scoring concept points at the EOS pop. With
 * EOS otherwise immediately firing, min_words=5 forces the loop to fall
 * back to non-EOS candidates for at least the first 5 steps.
 *--------------------------------------------------------------------------*/
static void test_min_words_suppresses_eos(void)
{
    static const char* names[6] = {"A", "B", "C", "D", "E", "EOS"};
    snn_language_bridge_t* b = build_n_words(6, names);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* Register pop 5 as EOS. */
    EXPECT(snn_language_bridge_set_eos_word_pop(b, 5) == 0, "set eos=5");

    /* min=5 with no max — the loop must emit at least 5 words before EOS
     * is allowed to fire. */
    EXPECT(snn_language_bridge_set_length_control(b, 5, 0) == 0,
           "set min=5 max=0");

    /* EOS (pop 5) is the highest-scoring concept; without suppression the
     * very first decode would pick EOS and produce returns -1. */
    float intent[6] = {0.30f, 0.25f, 0.20f, 0.15f, 0.10f, 1.00f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 6, &res);
    EXPECT(rc == 0 && res.text, "produce rc=%d (expected success with suppression)", rc);

    if (rc == 0 && res.text) {
        uint32_t n = count_words(res.text);
        EXPECT(n >= 5,
               "min_words=5 should yield ≥5 words; got %u text='%s'",
               n, res.text);
        /* EOS form should not appear in the output. */
        EXPECT(strstr(res.text, "EOS") == NULL,
               "EOS form must not appear in output; got '%s'", res.text);
    }

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.length_min_suppressions >= 1,
           "length_min_suppressions should advance ≥1; got %llu",
           (unsigned long long)s.length_min_suppressions);

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

/*--------------------------------------------------------------------------
 * Test 4: setter + getter validation.
 *--------------------------------------------------------------------------*/
static void test_setter_validation(void)
{
    static const char* names[3] = {"x", "y", "z"};
    snn_language_bridge_t* b = build_n_words(3, names);
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* NULL bridge rejected. */
    EXPECT(snn_language_bridge_set_length_control(NULL, 1, 2) == -1,
           "NULL bridge → -1");
    EXPECT(snn_language_bridge_get_length_control(NULL, NULL, NULL) == -1,
           "NULL bridge get → -1");

    /* min > max with both nonzero rejected. Config must stay unchanged. */
    EXPECT(snn_language_bridge_set_length_control(b, 7, 3) == -1,
           "min=7 max=3 → -1");
    uint32_t got_min = 99, got_max = 99;
    EXPECT(snn_language_bridge_get_length_control(b, &got_min, &got_max) == 0,
           "get after rejected set");
    EXPECT(got_min == 0 && got_max == 0,
           "config unchanged after rejection: min=%u max=%u", got_min, got_max);

    /* min == max accepted (boundary). */
    EXPECT(snn_language_bridge_set_length_control(b, 4, 4) == 0,
           "min==max=4 accepted");
    snn_language_bridge_get_length_control(b, &got_min, &got_max);
    EXPECT(got_min == 4 && got_max == 4,
           "min==max round-trip: %u/%u", got_min, got_max);

    /* min < max accepted. */
    EXPECT(snn_language_bridge_set_length_control(b, 2, 8) == 0,
           "min=2 max=8 accepted");
    snn_language_bridge_get_length_control(b, &got_min, &got_max);
    EXPECT(got_min == 2 && got_max == 8, "min<max round-trip");

    /* Either side disabled (0) bypasses cross-check — even with a tiny
     * max paired with a "high" min sentinel-0. */
    EXPECT(snn_language_bridge_set_length_control(b, 0, 1) == 0,
           "min=0 (disabled) max=1 accepted");
    snn_language_bridge_get_length_control(b, &got_min, &got_max);
    EXPECT(got_min == 0 && got_max == 1, "got min=%u max=%u", got_min, got_max);

    EXPECT(snn_language_bridge_set_length_control(b, 1, 0) == 0,
           "min=1 max=0 (disabled) accepted");
    snn_language_bridge_get_length_control(b, &got_min, &got_max);
    EXPECT(got_min == 1 && got_max == 0, "got min=%u max=%u", got_min, got_max);

    /* Above-cap values silently clamped to 1024. */
    EXPECT(snn_language_bridge_set_length_control(b, 99999, 99999) == 0,
           "huge values clamped");
    snn_language_bridge_get_length_control(b, &got_min, &got_max);
    EXPECT(got_min == 1024 && got_max == 1024,
           "clamped to 1024: got min=%u max=%u", got_min, got_max);

    /* Getter tolerates NULL out-pointers individually. */
    EXPECT(snn_language_bridge_get_length_control(b, NULL, &got_max) == 0,
           "NULL min ptr ok");
    EXPECT(got_max == 1024, "max still reads back");
    EXPECT(snn_language_bridge_get_length_control(b, &got_min, NULL) == 0,
           "NULL max ptr ok");
    EXPECT(got_min == 1024, "min still reads back");

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "[TB-7] test_lang_length_control\n");
    test_default_off_no_regression();
    test_max_words_caps_output();
    test_min_words_suppresses_eos();
    test_setter_validation();

    if (g_failures == 0) {
        fprintf(stderr, "OK — all 4 tests passed\n");
        return 0;
    }
    fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
    return 1;
}
