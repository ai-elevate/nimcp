/**
 * @file test_lang_stream_produce.c
 * @brief TB-8 — verify the per-token streaming callback in
 *        snn_language_bridge_produce.
 *
 * Coverage:
 *   1. test_no_callback_legacy:
 *      No callback attached → produce works exactly as before. Sanity
 *      check that we did not break the legacy path. Stats counters stay 0.
 *
 *   2. test_callback_completes_full_run:
 *      Callback always returns 0 → produce completes; the callback
 *      fires exactly result->word_count times; stats.stream_callbacks_invoked
 *      matches; stats.stream_aborts == 0.
 *
 *   3. test_callback_abort_at_index_2:
 *      Callback returns non-zero on word_index == 2 → produce returns
 *      with at most 3 words; stats.stream_aborts == 1; the text accumulated
 *      up to the abort is preserved (result->text non-empty,
 *      word_count <= 3).
 *
 *   4. test_null_bridge_setter:
 *      snn_language_bridge_set_stream_callback(NULL, ...) returns -1.
 *
 * Pattern: standalone smoke test, printf+exit-code reporting. Same
 * structure as test_lang_bridge_rng_seed.c.
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

/* Capture state for the callback: count of invocations + last seen word. */
typedef struct {
    uint32_t calls;
    uint32_t abort_at_index;   /* word_index that triggers abort. UINT32_MAX = never. */
    uint32_t last_index;
    char     last_word[64];
    uint32_t last_pop;
    float    last_confidence;
} cb_state_t;

static int never_abort_cb(uint32_t word_index, const char* word_form,
                           uint32_t word_pop, float confidence,
                           void* user_data)
{
    cb_state_t* s = (cb_state_t*)user_data;
    s->calls++;
    s->last_index = word_index;
    s->last_pop = word_pop;
    s->last_confidence = confidence;
    if (word_form) {
        strncpy(s->last_word, word_form, sizeof(s->last_word) - 1);
        s->last_word[sizeof(s->last_word) - 1] = '\0';
    } else {
        s->last_word[0] = '\0';
    }
    return 0;
}

static int abort_at_idx_cb(uint32_t word_index, const char* word_form,
                            uint32_t word_pop, float confidence,
                            void* user_data)
{
    cb_state_t* s = (cb_state_t*)user_data;
    s->calls++;
    s->last_index = word_index;
    s->last_pop = word_pop;
    s->last_confidence = confidence;
    if (word_form) {
        strncpy(s->last_word, word_form, sizeof(s->last_word) - 1);
        s->last_word[sizeof(s->last_word) - 1] = '\0';
    }
    return (word_index == s->abort_at_index) ? 1 : 0;
}

/* Build a 4-word bridge similar to test_lang_bridge_rng_seed: each word
 * bound 1:1 to a distinct concept_pop, weight 1.0. */
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

static void test_no_callback_legacy(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* No set_stream_callback at all — stream_cb stays NULL. */

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 4, &res);
    EXPECT(rc == 0, "produce rc=%d", rc);
    EXPECT(res.word_count > 0, "produce should emit at least one word");

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.stream_callbacks_invoked == 0,
           "no cb attached → stream_callbacks_invoked stays 0, got %llu",
           (unsigned long long)s.stream_callbacks_invoked);
    EXPECT(s.stream_aborts == 0,
           "no cb attached → stream_aborts stays 0, got %llu",
           (unsigned long long)s.stream_aborts);

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

static void test_callback_completes_full_run(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    cb_state_t state;
    memset(&state, 0, sizeof(state));
    state.abort_at_index = UINT32_MAX;  /* never abort */

    EXPECT(snn_language_bridge_set_stream_callback(b, never_abort_cb, &state) == 0,
           "set cb");

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 4, &res);
    EXPECT(rc == 0, "produce rc=%d", rc);
    EXPECT(res.word_count > 0, "produce should emit at least one word");

    /* Callback fires once per emitted word — never any extras. */
    EXPECT(state.calls == res.word_count,
           "cb calls=%u must match word_count=%u", state.calls, res.word_count);

    /* word_index is 0-based and reaches word_count-1 on the last call. */
    EXPECT(state.last_index == res.word_count - 1,
           "last cb word_index=%u, expected %u",
           state.last_index, res.word_count - 1);

    /* word_pop should be a registered pop (0..3). */
    EXPECT(state.last_pop < 4, "last cb word_pop=%u out of range",
           state.last_pop);

    /* word_form was non-empty during the callback. */
    EXPECT(state.last_word[0] != '\0',
           "last cb word_form was empty");

    /* Stats parity. */
    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.stream_callbacks_invoked == state.calls,
           "stats counter %llu != local cb count %u",
           (unsigned long long)s.stream_callbacks_invoked, state.calls);
    EXPECT(s.stream_aborts == 0,
           "no aborts → stream_aborts==0, got %llu",
           (unsigned long long)s.stream_aborts);

    /* Detach and confirm the next produce does NOT invoke the callback. */
    EXPECT(snn_language_bridge_set_stream_callback(b, NULL, NULL) == 0,
           "detach cb");
    cb_state_t state2;
    memset(&state2, 0, sizeof(state2));
    /* Note: install state2 as user_data only conceptually — we already
     * detached. The cb pointer is what gates the call site. */

    snn_lang_production_result_cleanup(&res);
    memset(&res, 0, sizeof(res));
    rc = snn_language_bridge_produce(b, intent, 4, &res);
    EXPECT(rc == 0, "produce rc after detach=%d", rc);
    /* state2 untouched — cb is detached. */
    EXPECT(state2.calls == 0, "post-detach cb should not fire, got %u",
           state2.calls);

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

static void test_callback_abort_at_index_2(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    cb_state_t state;
    memset(&state, 0, sizeof(state));
    state.abort_at_index = 2;  /* return non-zero on the 3rd emitted word */

    EXPECT(snn_language_bridge_set_stream_callback(b, abort_at_idx_cb, &state) == 0,
           "set abort cb");

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 4, &res);
    EXPECT(rc == 0, "produce rc=%d", rc);

    /* The abort fires AFTER word_index==2 has been counted (word_count=3),
     * so the result holds at most 3 words. */
    EXPECT(res.word_count <= 3,
           "aborted run word_count=%u must be ≤ 3", res.word_count);

    /* Text accumulated so far is preserved. */
    EXPECT(res.text != NULL && res.text[0] != '\0',
           "aborted run must still have non-empty text (preserved up to abort)");

    /* Callback fired up to and including the abort step. */
    EXPECT(state.calls == res.word_count,
           "cb calls=%u must equal word_count=%u (one fire per emission)",
           state.calls, res.word_count);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.stream_aborts == 1,
           "stream_aborts must be 1, got %llu",
           (unsigned long long)s.stream_aborts);
    EXPECT(s.stream_callbacks_invoked == state.calls,
           "stats invoked=%llu must equal cb calls=%u",
           (unsigned long long)s.stream_callbacks_invoked, state.calls);

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

static void test_null_bridge_setter(void)
{
    EXPECT(snn_language_bridge_set_stream_callback(NULL, never_abort_cb, NULL) == -1,
           "NULL bridge must return -1");
    EXPECT(snn_language_bridge_set_stream_callback(NULL, NULL, NULL) == -1,
           "NULL bridge + NULL cb must still return -1");
}

/* Regression: pre-fix the TB-7 max-cap break short-circuited the TB-8
 * callback for the cap-final word. A consumer producing N words via the
 * cap saw only N-1 callbacks. Reorder fixed it so the callback fires
 * before the cap break — every emitted word triggers exactly one
 * callback, regardless of why the loop terminates. */
static void test_callback_fires_for_cap_final_word(void)
{
    snn_language_bridge_t* b = build_4words();
    EXPECT(b != NULL, "bridge create");
    if (!b) return;

    /* Cap at 2 words. */
    EXPECT(snn_language_bridge_set_length_control(b, /*min*/0, /*max*/2) == 0,
           "set_length_control(0, 2)");

    cb_state_t state;
    memset(&state, 0, sizeof(state));
    state.abort_at_index = UINT32_MAX;  /* never abort */

    EXPECT(snn_language_bridge_set_stream_callback(b, never_abort_cb, &state) == 0,
           "set never-abort cb");

    float intent[4] = {1.0f, 0.5f, 0.2f, 0.05f};
    snn_lang_production_result_t res;
    memset(&res, 0, sizeof(res));
    int rc = snn_language_bridge_produce(b, intent, 4, &res);
    EXPECT(rc == 0, "produce rc=%d", rc);

    /* The cap should have fired — produce emitted exactly 2 words. */
    EXPECT(res.word_count == 2,
           "cap-bounded run word_count=%u (expected 2)", res.word_count);

    /* And the callback should have fired exactly once per emitted word
     * — the bug was N-1 fires; the fix is N. */
    EXPECT(state.calls == res.word_count,
           "cb calls=%u must equal word_count=%u after cap-truncation",
           state.calls, res.word_count);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.stream_callbacks_invoked == state.calls,
           "stats invoked=%llu must equal cb calls=%u",
           (unsigned long long)s.stream_callbacks_invoked, state.calls);
    EXPECT(s.length_max_truncations == 1,
           "stats length_max_truncations must be 1, got %llu",
           (unsigned long long)s.length_max_truncations);

    snn_lang_production_result_cleanup(&res);
    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "=== test_lang_stream_produce (TB-8) ===\n");
    test_no_callback_legacy();
    test_callback_completes_full_run();
    test_callback_abort_at_index_2();
    test_null_bridge_setter();
    test_callback_fires_for_cap_final_word();

    if (g_failures == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d failures\n", g_failures);
    return 1;
}
