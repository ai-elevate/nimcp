/**
 * @file test_snn_negative_da.c
 * @brief Slice F — Anti-Hebbian punishment via negative DA modulation gain.
 *
 * Real neuromodulation supports punishment as well as reward: tonic DA
 * baseline is non-zero, and DA dips below baseline implement aversive
 * learning. The brain's neuromodulator system clamps DA concentration to
 * [0, 1] (no negative concentrations), so we encode "negative DA" as a
 * negative `da_modulation_gain` on the bridge config — the three-factor
 * product `modulation = 1 + DA × gain` then becomes < 1 (depression) or
 * even < 0 (LTP→LTD sign flip) when |DA × gain| > 1.
 *
 * Coverage:
 *   1. test_clamp_widened:
 *      set_da_modulation_gain accepts values in [da_min, +200], rejects
 *      NaN, clamps out-of-range values.
 *
 *   2. test_positive_da_strengthens:
 *      Fire pre+post pair, DA=0.5, gain=+10 → modulation=+6 → weight up.
 *
 *   3. test_negative_gain_weakens:
 *      Same pre+post pair, DA=0.5, gain=-10 → modulation=-4 → weight down.
 *      Verifies the LTP sign flips to LTD under negative gain.
 *
 *   4. test_weight_floors_at_zero:
 *      Repeated pre+post pairs with negative gain → weight clamps at the
 *      binding floor (W_MIN, typically 0), never goes negative.
 *
 *   5. test_zero_da_no_change:
 *      DA=0, gain=-10 → modulation = 1 + 0 × (-10) = 1.0. No change either
 *      direction (the three-factor product is zero because DA is zero —
 *      reward-baseline alignment).
 *
 *   6. test_no_nan_on_negative:
 *      Drive a long sequence under negative gain → last_da_modulation
 *      remains finite, no NaN/Inf propagation.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"

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

static snn_lang_config_t make_cfg(void) {
    snn_lang_config_t c;
    memset(&c, 0, sizeof(c));
    c.max_concept_pops = 8;
    c.max_word_pops = 8;
    c.neurons_per_pop = 8;
    c.stdp_tau_plus = 20.0f;
    c.stdp_tau_minus = 20.0f;
    c.stdp_a_plus = 0.05f;
    c.stdp_a_minus = 0.025f;
    c.stdp_learning_rate = 0.1f;
    c.binding_w_max = 1.0f;
    c.decode_window_ms = 50.0f;
    c.decay_rate = 0.95f;
    c.spike_blend = 0.5f;
    c.enable_da_modulation = true;
    c.da_modulation_gain = 0.0f;     /* set per test */
    c.da_min = -200.0f;              /* Slice F: enable punishment range */
    c.activation_tau_ms = 50.0f;
    return c;
}

/* Fire one pre→post LTP pair at +2 ms (LTP window) on a fresh binding,
 * then return (avg_binding_weight, last_da_modulation) via output params.
 * The binding starts at seed_w and the bridge's STDP rule updates it
 * via standard soft-bounded STDP scaled by the DA modulation factor. */
static void drive_pair(snn_language_bridge_t* bridge,
                       uint32_t cpop, uint32_t wpop,
                       float seed_w,
                       float* out_avg_weight,
                       float* out_modulation)
{
    (void)snn_language_bridge_register_concept(bridge, cpop, 100u + cpop);
    (void)snn_language_bridge_register_word(bridge, wpop, "x");
    (void)snn_language_bridge_bind(bridge, cpop, wpop, seed_w);

    /* concept fires at 10ms, word at 12ms → +2ms LTP window. */
    (void)snn_language_bridge_concept_spike(bridge, cpop, 10.0f);
    (void)snn_language_bridge_word_spike(bridge, wpop, 12.0f);
    (void)snn_language_bridge_apply_stdp(bridge, 20.0f);

    snn_lang_stats_t s;
    (void)snn_language_bridge_get_stats(bridge, &s);
    if (out_avg_weight) *out_avg_weight = s.avg_binding_weight;
    if (out_modulation) *out_modulation = s.last_da_modulation;
}

static neuromodulator_system_t make_neuromod_with_da(float da_level)
{
    neuromodulator_config_t nc;
    memset(&nc, 0, sizeof(nc));
    neuromodulator_system_t sys = neuromodulator_system_create(&nc);
    if (!sys) return NULL;
    (void)neuromodulator_set_level(sys, NEUROMOD_DOPAMINE, da_level);
    return sys;
}

/* ============================================================
 * Test 1 — clamp widened to [da_min, +200]
 * ============================================================ */
static void test_clamp_widened(void)
{
    snn_lang_config_t cfg = make_cfg();
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    /* NaN rejected. */
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, NAN) == -1,
           "NaN rejected");

    /* Negative gain now ACCEPTED (was rejected pre-Slice-F). */
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, -10.0f) == 0,
           "negative gain accepted");
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, -200.0f) == 0,
           "gain at lower bound accepted");
    /* Below da_min → clamped, not rejected. */
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, -9999.0f) == 0,
           "below-min clamped, not rejected");

    /* Upper bound still 200 — clamp doesn't reject. */
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, 9999.0f) == 0,
           "above-max clamped, not rejected");

    /* Zero still legal (means "modulation off"). */
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, 0.0f) == 0,
           "0 gain accepted");

    snn_language_bridge_destroy(b);
}

/* ============================================================
 * Test 2 — positive DA strengthens (sanity baseline)
 * ============================================================ */
static void test_positive_da_strengthens(void)
{
    snn_lang_config_t cfg = make_cfg();
    cfg.da_modulation_gain = 10.0f;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    neuromodulator_system_t sys = make_neuromod_with_da(0.5f);
    EXPECT(sys != NULL, "neuromod create");
    if (!sys) { snn_language_bridge_destroy(b); return; }

    EXPECT(snn_language_bridge_connect_neuromod(b, sys) == 0,
           "connect neuromod");

    const float seed = 0.5f;
    float avg = 0.0f, mod = 0.0f;
    drive_pair(b, 0, 0, seed, &avg, &mod);

    /* Modulation = 1 + 0.5 * 10 = 6.0  (positive, > 1.0). */
    EXPECT(mod > 1.0f && mod < 100.0f,
           "positive DA modulation > 1.0, got %.4f", mod);
    /* avg_binding_weight should INCREASE above the seed. */
    EXPECT(avg > seed,
           "positive DA: weight should increase from %.3f, got %.4f",
           seed, avg);

    neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

/* ============================================================
 * Test 3 — negative gain → LTP sign flips → weight DECREASES
 * ============================================================ */
static void test_negative_gain_weakens(void)
{
    snn_lang_config_t cfg = make_cfg();
    cfg.da_modulation_gain = -10.0f;  /* punishment gain */
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    neuromodulator_system_t sys = make_neuromod_with_da(0.5f);
    EXPECT(sys != NULL, "neuromod create");
    if (!sys) { snn_language_bridge_destroy(b); return; }

    EXPECT(snn_language_bridge_connect_neuromod(b, sys) == 0,
           "connect neuromod");

    const float seed = 0.5f;
    float avg = 0.0f, mod = 0.0f;
    drive_pair(b, 0, 0, seed, &avg, &mod);

    /* Modulation = 1 + 0.5 * (-10) = -4.0. Negative → LTP becomes LTD. */
    EXPECT(mod < 0.0f,
           "negative DA modulation < 0, got %.4f", mod);
    /* avg_binding_weight should DECREASE below the seed (or hit floor). */
    EXPECT(avg < seed,
           "negative DA: weight should decrease from %.3f, got %.4f",
           seed, avg);
    /* Weight must never go below zero (binding W_MIN). */
    EXPECT(avg >= 0.0f,
           "negative DA: weight must clamp at W_MIN=0, got %.4f", avg);
    /* Must be finite. */
    EXPECT(isfinite(avg) && isfinite(mod),
           "negative DA: avg=%.4f mod=%.4f must both be finite", avg, mod);

    neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

/* ============================================================
 * Test 4 — repeated punishment → weight floors at W_MIN (0)
 * ============================================================ */
static void test_weight_floors_at_zero(void)
{
    snn_lang_config_t cfg = make_cfg();
    cfg.da_modulation_gain = -50.0f;  /* strong punishment */
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    neuromodulator_system_t sys = make_neuromod_with_da(1.0f);  /* max DA */
    EXPECT(sys != NULL, "neuromod create");
    if (!sys) { snn_language_bridge_destroy(b); return; }
    EXPECT(snn_language_bridge_connect_neuromod(b, sys) == 0,
           "connect neuromod");

    /* Seed one binding, then hammer it with negative-gain pre+post pairs.
     * Use repeated re-binds (the bridge takes max(old,new) on bind, but
     * the binding hash table preserves a single node so apply_stdp
     * keeps depressing the same weight). */
    const uint32_t cpop = 1, wpop = 1;
    (void)snn_language_bridge_register_concept(b, cpop, 100u);
    (void)snn_language_bridge_register_word(b, wpop, "x");
    (void)snn_language_bridge_bind(b, cpop, wpop, 0.5f);

    float final_avg = 0.5f;
    for (int i = 0; i < 50; i++) {
        (void)snn_language_bridge_concept_spike(b, cpop, 10.0f);
        (void)snn_language_bridge_word_spike(b, wpop, 12.0f);
        (void)snn_language_bridge_apply_stdp(b, 20.0f);

        snn_lang_stats_t s;
        (void)snn_language_bridge_get_stats(b, &s);
        final_avg = s.avg_binding_weight;

        /* Invariant: weight never goes below zero, never NaN/Inf. */
        EXPECT(isfinite(final_avg),
               "iter %d: weight must stay finite, got %.4f", i, final_avg);
        EXPECT(final_avg >= 0.0f,
               "iter %d: weight must clamp at W_MIN=0, got %.4f",
               i, final_avg);
    }

    /* After many punishments, weight should be at or near zero. */
    EXPECT(final_avg < 0.1f,
           "repeated punishment: weight should drop near floor, got %.4f",
           final_avg);

    neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

/* ============================================================
 * Test 5 — DA=0 → no change regardless of gain sign
 * ============================================================ */
static void test_zero_da_no_change(void)
{
    snn_lang_config_t cfg = make_cfg();
    cfg.da_modulation_gain = -10.0f;
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    /* DA explicitly zero. */
    neuromodulator_system_t sys = make_neuromod_with_da(0.0f);
    EXPECT(sys != NULL, "neuromod create");
    if (!sys) { snn_language_bridge_destroy(b); return; }
    EXPECT(snn_language_bridge_connect_neuromod(b, sys) == 0,
           "connect neuromod");

    float avg = 0.0f, mod = 0.0f;
    drive_pair(b, 0, 0, 0.5f, &avg, &mod);

    /* Modulation = 1 + 0 * (-10) = 1.0 exactly. */
    EXPECT(fabsf(mod - 1.0f) < 1e-6f,
           "DA=0 → modulation must be 1.0, got %.6f", mod);
    /* Weight changes ONLY by the baseline STDP delta (small positive).
     * The three-factor product is the unmodulated LTP delta. With
     * modulation=1 and seed=0.5 the weight should be > 0.5 by the
     * standard LTP amount and < 1.0. */
    EXPECT(avg > 0.5f && avg <= 1.0f,
           "DA=0: baseline LTP only, expected 0.5 < avg <= 1.0, got %.4f",
           avg);

    neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

/* ============================================================
 * Test 6 — long negative-DA run: no NaN/Inf
 * ============================================================ */
static void test_no_nan_on_negative(void)
{
    snn_lang_config_t cfg = make_cfg();
    cfg.da_modulation_gain = -100.0f;  /* extreme punishment */
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    neuromodulator_system_t sys = make_neuromod_with_da(1.0f);
    EXPECT(sys != NULL, "neuromod create");
    if (!sys) { snn_language_bridge_destroy(b); return; }
    EXPECT(snn_language_bridge_connect_neuromod(b, sys) == 0,
           "connect neuromod");

    /* Hammer with 200 pre+post pairs across multiple bindings to also
     * exercise the iterator + soft-bound path with negative modulation. */
    for (int i = 0; i < 200; i++) {
        uint32_t cpop = (uint32_t)(i % 4);
        uint32_t wpop = (uint32_t)(i % 4);
        (void)snn_language_bridge_register_concept(b, cpop, 100u + cpop);
        (void)snn_language_bridge_register_word(b, wpop, "x");
        (void)snn_language_bridge_bind(b, cpop, wpop, 0.5f);
        (void)snn_language_bridge_concept_spike(b, cpop, 10.0f);
        (void)snn_language_bridge_word_spike(b, wpop, 12.0f);
        (void)snn_language_bridge_apply_stdp(b, 20.0f);
    }

    snn_lang_stats_t s;
    (void)snn_language_bridge_get_stats(b, &s);
    EXPECT(isfinite(s.last_da_modulation),
           "last_da_modulation must remain finite, got %.4f",
           s.last_da_modulation);
    EXPECT(isfinite(s.avg_binding_weight),
           "avg_binding_weight must remain finite, got %.4f",
           s.avg_binding_weight);
    EXPECT(s.avg_binding_weight >= 0.0f,
           "weight must stay >= 0, got %.4f", s.avg_binding_weight);
    EXPECT(s.da_gated_stdp_passes >= 1,
           "negative-gain path must still bump da_gated_stdp_passes");

    neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "=== test_snn_negative_da (Slice F) ===\n");
    test_clamp_widened();
    test_positive_da_strengthens();
    test_negative_gain_weakens();
    test_weight_floors_at_zero();
    test_zero_da_no_change();
    test_no_nan_on_negative();

    if (g_failures == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d failures\n", g_failures);
    return 1;
}
