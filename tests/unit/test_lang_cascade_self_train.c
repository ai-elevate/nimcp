/**
 * @file test_lang_cascade_self_train.c
 * @brief Wave 2 Item #10 — verify cascade self-train reward closes the loop.
 *
 * Stage 11 of the production cascade reads state->self_match and applies
 * a reward-modulated update to the SNN language bridge via
 * snn_language_bridge_echo_correct(). This file exercises the pure helper
 * cascade_apply_self_train_reward() (no brain needed) and confirms:
 *
 *  1. test_skip_on_null_bridge:
 *      Helper short-circuits cleanly when the bridge is NULL.
 *  2. test_skip_on_empty_utterance:
 *      Empty / whitespace-only utterance is a no-op (returns 0, out_reward=0).
 *  3. test_skip_on_invalid_self_match:
 *      NaN, Inf, or out-of-range self_match returns 0 with out_reward=0.
 *  4. test_positive_reward_strengthens_bindings:
 *      Positive (self_match - baseline) drives at least one binding
 *      strengthening across produced words.
 *  5. test_zero_reward_no_update:
 *      When self_match exactly matches baseline, reward is 0 and the
 *      bridge is not touched (echo_correct rejects lr_scale=0).
 *  6. test_baseline_ema_updates:
 *      Repeated calls with constant self_match drift the baseline
 *      monotonically toward self_match (EMA convergence).
 *  7. test_negative_reward_is_attenuating_not_destructive:
 *      A self_match below baseline produces a negative train_reward
 *      diagnostic but still calls echo_correct with |reward| × lr_scale,
 *      so the bindings still receive an LTP (smaller magnitude) — the
 *      sign IS preserved in out_reward for the trainer to act on.
 *  8. test_tunable_clamps:
 *      alpha outside [0,1] and lr_scale outside [-10,10] are clamped
 *      and the helper completes without crashing.
 */

#include "language/nimcp_communication_cascade.h"
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

static snn_lang_config_t default_cfg(void) {
    snn_lang_config_t c;
    memset(&c, 0, sizeof(c));
    c.max_concept_pops = 16;
    /* echo_correct hashes form_hash % SNN_LANG_MAX_WORD_POPS (32768) — to
     * keep arbitrary test words addressable we size max_word_pops to the
     * full capacity. Otherwise most hashes land above word_pops_capacity
     * and echo_correct returns -2 (target word not registered). */
    c.max_word_pops = 32768;  /* SNN_LANG_MAX_WORD_POPS */
    c.neurons_per_pop = 8;
    c.stdp_tau_plus = 20.0f;
    c.stdp_tau_minus = 20.0f;
    /* Large a_plus so the per-call delta is large enough to detect. */
    c.stdp_a_plus = 0.5f;
    c.stdp_a_minus = 0.25f;
    c.stdp_learning_rate = 0.5f;
    c.binding_w_max = 1.0f;
    c.decode_window_ms = 50.0f;
    c.decay_rate = 0.95f;
    c.spike_blend = 0.5f;
    c.enable_da_modulation = false;
    c.da_modulation_gain = 0.0f;
    c.activation_tau_ms = 50.0f;
    /* Skip the comprehend-stdp gate (CSTDP) on echo_correct — echo_correct
     * doesn't gate on the comprehend_stdp_min_activation but it does read
     * stdp_a_plus, so we just need sane LTP knobs. */
    c.enable_comprehend_stdp = false;
    return c;
}

/* Construct a bridge with enough concepts/words registered to exercise
 * the strengthen path. Returns NULL on alloc failure. */
static snn_language_bridge_t* make_bridge_with_words(const char** words,
                                                       uint32_t n_words) {
    snn_lang_config_t cfg = default_cfg();
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    if (!b) return NULL;
    /* Register a handful of concept pops so the bridge has at least
     * 16 dimensions to receive activation across. */
    for (uint32_t c = 0; c < 16; c++) {
        (void)snn_language_bridge_register_concept(b, c, 1000u + c);
    }
    /* echo_correct hashes the word form modulo max_word_pops; pre-register
     * each test word at its hashed slot so the binding lands at a known
     * pop index. The helper registers it again internally — idempotent. */
    for (uint32_t i = 0; i < n_words; i++) {
        /* FNV-1a, lowercase — duplicates the bridge-internal hash. */
        uint32_t h = 2166136261u;
        for (const char* p = words[i]; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
            h ^= c;
            h *= 16777619u;
        }
        uint32_t pop = h % 32768u;   /* SNN_LANG_MAX_WORD_POPS */
        (void)snn_language_bridge_register_word(b, pop, words[i]);
    }
    return b;
}

/* A reproducible activation vector — first N entries positive, rest zero. */
static void make_intent(float* intent, uint32_t dim, uint32_t n_active) {
    for (uint32_t i = 0; i < dim; i++) {
        intent[i] = (i < n_active) ? (0.5f + 0.05f * (float)i) : 0.0f;
    }
}

/*--------------------------------------------------------------------------*/

static void test_skip_on_null_bridge(void)
{
    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.0f;
    float reward = 12345.0f;  /* sentinel — should be reset to 0 */
    int rc = cascade_apply_self_train_reward(
                NULL, intent, 16, "hello world", 0.8f,
                &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "NULL bridge → rc=0, got %d", rc);
    EXPECT(reward == 0.0f, "NULL bridge → reward=0, got %.4f", reward);
    EXPECT(baseline == 0.0f, "NULL bridge → baseline unchanged, got %.4f",
           baseline);
}

static void test_skip_on_empty_utterance(void)
{
    const char* words[] = { "hello" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 1);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.0f;
    float reward = 1.0f;

    int rc = cascade_apply_self_train_reward(
                b, intent, 16, "", 0.8f,
                &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "empty utterance → rc=0, got %d", rc);
    EXPECT(reward == 0.0f, "empty utterance → reward=0, got %.4f", reward);

    rc = cascade_apply_self_train_reward(
            b, intent, 16, NULL, 0.8f,
            &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "NULL utterance → rc=0, got %d", rc);
    EXPECT(reward == 0.0f, "NULL utterance → reward=0, got %.4f", reward);

    snn_language_bridge_destroy(b);
}

static void test_skip_on_invalid_self_match(void)
{
    const char* words[] = { "hello" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 1);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.0f;
    float reward = 0.0f;

    /* NaN self_match. */
    int rc = cascade_apply_self_train_reward(
                b, intent, 16, "hello", NAN,
                &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "NaN self_match → rc=0, got %d", rc);
    EXPECT(reward == 0.0f, "NaN self_match → reward=0, got %.4f", reward);

    /* +Inf self_match. */
    rc = cascade_apply_self_train_reward(
            b, intent, 16, "hello", INFINITY,
            &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "+Inf self_match → rc=0, got %d", rc);

    /* Out of range. */
    rc = cascade_apply_self_train_reward(
            b, intent, 16, "hello", 1.5f,
            &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "self_match>1 → rc=0, got %d", rc);

    rc = cascade_apply_self_train_reward(
            b, intent, 16, "hello", -0.1f,
            &baseline, 0.05f, 1.0f, &reward);
    EXPECT(rc == 0, "self_match<0 → rc=0, got %d", rc);

    snn_language_bridge_destroy(b);
}

static void test_positive_reward_strengthens_bindings(void)
{
    const char* words[] = { "cat", "dog" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 2);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.0f;          /* fresh — first call rewards = self_match */
    float reward_out = 0.0f;

    /* self_match well above baseline → strong positive reward. */
    int n = cascade_apply_self_train_reward(
                b, intent, 16, "cat dog", 0.9f,
                &baseline, 0.05f, 1.0f, &reward_out);

    EXPECT(n > 0, "positive reward → bindings strengthened, got n=%d", n);
    EXPECT(fabsf(reward_out - 0.9f) < 1e-5f,
           "reward = self_match - baseline = 0.9, got %.4f", reward_out);
    /* Baseline EMA updated: 0 + 0.05*0.9 = 0.045. */
    EXPECT(fabsf(baseline - 0.045f) < 1e-5f,
           "baseline EMA → 0.045, got %.4f", baseline);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.echo_correct_calls >= 2,
           "echo_correct fired per word, got %llu",
           (unsigned long long)s.echo_correct_calls);

    snn_language_bridge_destroy(b);
}

static void test_zero_reward_no_update(void)
{
    const char* words[] = { "cat" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 1);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.6f;
    float reward_out = 99.0f;

    /* self_match == baseline → reward is exactly 0. */
    int n = cascade_apply_self_train_reward(
                b, intent, 16, "cat", 0.6f,
                &baseline, 0.05f, 1.0f, &reward_out);

    EXPECT(n == 0, "zero reward → 0 bindings, got n=%d", n);
    EXPECT(reward_out == 0.0f, "reward should be 0, got %.4f", reward_out);
    /* Baseline still EMA-updated even at zero reward. */
    EXPECT(fabsf(baseline - 0.6f) < 1e-5f,
           "baseline unchanged when self_match==baseline, got %.4f", baseline);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.echo_correct_calls == 0,
           "echo_correct not called on zero reward, got %llu",
           (unsigned long long)s.echo_correct_calls);

    snn_language_bridge_destroy(b);
}

static void test_baseline_ema_updates(void)
{
    /* No bridge needed — exercise EMA in isolation by passing NULL bridge
     * is not enough because the helper short-circuits before the EMA
     * update on NULL bridge. Use a real bridge with constant self_match
     * and watch the baseline asymptote. */
    const char* words[] = { "hi" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 1);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.0f;
    float reward_out = 0.0f;
    const float target = 0.8f;
    const float alpha = 0.1f;

    for (int iter = 0; iter < 50; iter++) {
        cascade_apply_self_train_reward(
            b, intent, 16, "hi", target,
            &baseline, alpha, 1.0f, &reward_out);
    }

    /* After 50 EMA steps with alpha=0.1, baseline should be very close to
     * the target value. Theoretical decay: residual = (1-alpha)^50. */
    float residual = powf(1.0f - alpha, 50.0f);
    EXPECT(fabsf(baseline - target) < 2.0f * residual,
           "baseline EMA → target=%.4f, residual bound=%.4f, got %.4f",
           target, residual, baseline);
    /* And the reward signal should have shrunk toward 0. */
    EXPECT(fabsf(reward_out) < 0.05f,
           "reward → ~0 after EMA convergence, got %.4f", reward_out);

    snn_language_bridge_destroy(b);
}

static void test_negative_reward_attenuates_not_destructive(void)
{
    const char* words[] = { "ugh" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 1);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.7f;
    float reward_out = 99.0f;

    /* self_match BELOW baseline → negative reward. */
    int n = cascade_apply_self_train_reward(
                b, intent, 16, "ugh", 0.2f,
                &baseline, 0.05f, 1.0f, &reward_out);

    /* Sign preserved in diagnostic — trainer can act on it. */
    EXPECT(reward_out < 0.0f,
           "negative reward preserved in out_reward, got %.4f", reward_out);
    EXPECT(fabsf(reward_out - (-0.5f)) < 1e-5f,
           "reward = 0.2 - 0.7 = -0.5, got %.4f", reward_out);

    /* Helper still calls echo_correct with |reward| × lr_scale, so n
     * may be positive (attenuated LTP rather than reversed LTD). The
     * important property: it does NOT segfault and the count is
     * non-negative. */
    EXPECT(n >= 0, "negative reward → non-negative binding count, got %d", n);

    snn_language_bridge_destroy(b);
}

static void test_tunable_clamps(void)
{
    const char* words[] = { "x" };
    snn_language_bridge_t* b = make_bridge_with_words(words, 1);
    EXPECT(b != NULL, "create");
    if (!b) return;

    float intent[16];
    make_intent(intent, 16, 4);
    float baseline = 0.0f;
    float reward_out = 0.0f;

    /* alpha = NaN → clamped to 0 (no EMA update). */
    cascade_apply_self_train_reward(
        b, intent, 16, "x", 0.5f,
        &baseline, NAN, 1.0f, &reward_out);
    EXPECT(baseline == 0.0f, "NaN alpha → baseline frozen, got %.4f", baseline);

    /* alpha = 5.0 → clamped to 1.0 (baseline becomes self_match). */
    baseline = 0.0f;
    cascade_apply_self_train_reward(
        b, intent, 16, "x", 0.5f,
        &baseline, 5.0f, 1.0f, &reward_out);
    EXPECT(fabsf(baseline - 0.5f) < 1e-5f,
           "alpha>1 → clamped to 1, baseline=self_match=0.5, got %.4f",
           baseline);

    /* alpha = -1.0 → clamped to 0. */
    baseline = 0.3f;
    cascade_apply_self_train_reward(
        b, intent, 16, "x", 0.5f,
        &baseline, -1.0f, 1.0f, &reward_out);
    EXPECT(fabsf(baseline - 0.3f) < 1e-5f,
           "alpha<0 → clamped to 0, baseline unchanged, got %.4f", baseline);

    /* lr_scale = 1000 → clamped, no crash. */
    baseline = 0.0f;
    int n = cascade_apply_self_train_reward(
                b, intent, 16, "x", 0.5f,
                &baseline, 0.05f, 1000.0f, &reward_out);
    EXPECT(n >= 0, "huge lr_scale → completes, got n=%d", n);

    /* lr_scale = -100 → clamped to -10. echo_correct rejects negative
     * lr_scale so the helper applies fabsf and returns 0 for those words
     * where echo_correct says no, but does not crash. */
    baseline = 0.0f;
    n = cascade_apply_self_train_reward(
            b, intent, 16, "x", 0.5f,
            &baseline, 0.05f, -100.0f, &reward_out);
    EXPECT(n >= 0, "negative lr_scale → completes, got n=%d", n);

    snn_language_bridge_destroy(b);
}

/*--------------------------------------------------------------------------*/

int main(void)
{
    fprintf(stderr, "=== test_lang_cascade_self_train (Wave 2 Item #10) ===\n");
    test_skip_on_null_bridge();
    test_skip_on_empty_utterance();
    test_skip_on_invalid_self_match();
    test_positive_reward_strengthens_bindings();
    test_zero_reward_no_update();
    test_baseline_ema_updates();
    test_negative_reward_attenuates_not_destructive();
    test_tunable_clamps();

    if (g_failures == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d failures\n", g_failures);
    return 1;
}
