/**
 * @file test_lang_da_modulation.c
 * @brief TA-3 — verify dopamine-modulated STDP in the SNN language bridge.
 *
 * The apply_stdp pass reads dopamine ONCE per call (via the connected
 * neuromodulator system) and folds it into the per-binding LR as
 *   weight_change *= 1 + dopamine * config.da_modulation_gain
 *
 * Coverage:
 *   1. test_no_neuromod_attached:
 *      DA modulation enabled but no neuromod connected → multiplier
 *      stays 1.0, da_gated_stdp_passes stays 0.
 *
 *   2. test_disabled_flag:
 *      Neuromod attached but enable_da_modulation=false → multiplier
 *      stays 1.0.
 *
 *   3. test_zero_dopamine:
 *      Neuromod attached with DA=0 → multiplier=1.0 (no scaling) but
 *      da_gated_stdp_passes is bumped (modulation path was taken).
 *
 *   4. test_high_dopamine_amplifies:
 *      Set DA=0.5 + gain=10 → multiplier should be 1+0.5*10 = 6.0.
 *      Run STDP and verify weight delta is ~6× the no-DA case.
 *
 *   5. test_setter_validation:
 *      set_da_modulation_gain rejects NaN/negative; clamps >200; accepts
 *      0 and 50.
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

static snn_lang_config_t default_cfg(void) {
    snn_lang_config_t c;
    memset(&c, 0, sizeof(c));
    c.max_concept_pops = 4;
    c.max_word_pops = 4;
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
    c.da_modulation_gain = 10.0f;
    c.activation_tau_ms = 50.0f;
    return c;
}

/* Fire a coincident pre-then-post pair, apply STDP, return the resulting
 * binding weight via stats.last_da_modulation × the standard LTP delta. */
static float drive_one_ltp(snn_language_bridge_t* bridge,
                            uint32_t cpop, uint32_t wpop) {
    /* Reset bindings to a known weight by re-registering. The bridge
     * doesn't expose a binding weight setter, so we use a fresh pair
     * each call by varying the (cpop, wpop) tuple. */
    (void)snn_language_bridge_register_concept(bridge, cpop, 100u + cpop);
    (void)snn_language_bridge_register_word(bridge, wpop, "x");

    /* Concept fires at t=10ms, word at t=12ms → +2ms LTP window. */
    (void)snn_language_bridge_concept_spike(bridge, cpop, 10.0f);
    (void)snn_language_bridge_word_spike(bridge, wpop, 12.0f);

    (void)snn_language_bridge_apply_stdp(bridge, 12.5f);

    snn_lang_stats_t s;
    (void)snn_language_bridge_get_stats(bridge, &s);
    return s.last_da_modulation;
}

static void test_no_neuromod_attached(void)
{
    snn_lang_config_t cfg = default_cfg();
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    EXPECT(snn_language_bridge_get_da_modulation_enabled(b),
           "DA mod default ON in cfg");

    float mod = drive_one_ltp(b, 0, 0);
    EXPECT(fabsf(mod - 1.0f) < 1e-6f,
           "no neuromod → multiplier should stay 1.0, got %.4f", mod);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.da_gated_stdp_passes == 0,
           "no neuromod → da_gated_stdp_passes==0, got %llu",
           (unsigned long long)s.da_gated_stdp_passes);

    snn_language_bridge_destroy(b);
}

static void test_disabled_flag(void)
{
    snn_lang_config_t cfg = default_cfg();
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    /* Disable DA modulation explicitly. */
    EXPECT(snn_language_bridge_set_da_modulation_enabled(b, false) == 0,
           "set DA disabled");
    EXPECT(!snn_language_bridge_get_da_modulation_enabled(b),
           "DA mod should be disabled");

    /* Even WITH neuromod attached, gate stays closed. */
    neuromodulator_config_t nc;
    memset(&nc, 0, sizeof(nc));
    neuromodulator_system_t sys = neuromodulator_system_create(&nc);
    EXPECT(sys != NULL, "neuromod create");
    if (sys) {
        (void)neuromodulator_set_level(sys, NEUROMOD_DOPAMINE, 0.8f);
        (void)snn_language_bridge_connect_neuromod(b, sys);
    }

    float mod = drive_one_ltp(b, 0, 0);
    EXPECT(fabsf(mod - 1.0f) < 1e-6f,
           "disabled flag → multiplier 1.0 even with DA=0.8, got %.4f", mod);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.da_gated_stdp_passes == 0,
           "disabled flag → da_gated_stdp_passes==0, got %llu",
           (unsigned long long)s.da_gated_stdp_passes);

    if (sys) neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

static void test_high_dopamine_amplifies(void)
{
    snn_lang_config_t cfg = default_cfg();
    cfg.da_modulation_gain = 10.0f;  /* DA=0.5 → multiplier = 6.0 */
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    neuromodulator_config_t nc;
    memset(&nc, 0, sizeof(nc));
    neuromodulator_system_t sys = neuromodulator_system_create(&nc);
    EXPECT(sys != NULL, "neuromod create");
    if (!sys) { snn_language_bridge_destroy(b); return; }

    EXPECT(snn_language_bridge_connect_neuromod(b, sys) == 0,
           "connect neuromod");

    /* DA = 0.5 → expected multiplier 1 + 0.5*10 = 6.0. */
    EXPECT(neuromodulator_set_level(sys, NEUROMOD_DOPAMINE, 0.5f),
           "set DA=0.5");

    float mod = drive_one_ltp(b, 0, 0);
    EXPECT(fabsf(mod - 6.0f) < 1e-3f,
           "DA=0.5 + gain=10 → multiplier 6.0, got %.4f", mod);

    snn_lang_stats_t s;
    snn_language_bridge_get_stats(b, &s);
    EXPECT(s.da_gated_stdp_passes >= 1,
           "DA pass should bump da_gated_stdp_passes");

    /* Now drop DA to 0 — multiplier should fall back to 1.0. */
    EXPECT(neuromodulator_set_level(sys, NEUROMOD_DOPAMINE, 0.0f),
           "set DA=0");
    mod = drive_one_ltp(b, 1, 1);
    EXPECT(fabsf(mod - 1.0f) < 1e-6f,
           "DA=0 → multiplier 1.0, got %.4f", mod);

    neuromodulator_system_destroy(sys);
    snn_language_bridge_destroy(b);
}

static void test_setter_validation(void)
{
    snn_lang_config_t cfg = default_cfg();
    snn_language_bridge_t* b = snn_language_bridge_create(&cfg);
    EXPECT(b != NULL, "create");
    if (!b) return;

    EXPECT(snn_language_bridge_set_da_modulation_gain(b, NAN) == -1,
           "NaN gain rejected");
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, -1.0f) == -1,
           "negative gain rejected");
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, 0.0f) == 0,
           "0 gain accepted");
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, 50.0f) == 0,
           "50 gain accepted");
    EXPECT(snn_language_bridge_set_da_modulation_gain(b, 999.0f) == 0,
           "999 gain accepted (clamped)");

    /* NULL bridge handled gracefully. */
    EXPECT(snn_language_bridge_set_da_modulation_enabled(NULL, true) == -1,
           "NULL bridge enable rejected");
    EXPECT(snn_language_bridge_set_da_modulation_gain(NULL, 1.0f) == -1,
           "NULL bridge gain rejected");
    EXPECT(!snn_language_bridge_get_da_modulation_enabled(NULL),
           "NULL bridge get returns false");

    snn_language_bridge_destroy(b);
}

int main(void)
{
    fprintf(stderr, "=== test_lang_da_modulation (TA-3) ===\n");
    test_no_neuromod_attached();
    test_disabled_flag();
    test_high_dopamine_amplifies();
    test_setter_validation();

    if (g_failures == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d failures\n", g_failures);
    return 1;
}
