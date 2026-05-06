/**
 * @file test_snn_lang_bridge_stdp.c
 * @brief D1 — Verify spike-driven STDP fires when grounded_language mirrors
 *        a word↔concept binding into the SNN language bridge.
 *
 * Pattern: standalone smoke test (no GTest dep), matches the style of
 * test_bulk_lexicon.c / test_gl_legacy_skip.c. Compile:
 *   gcc -I include tests/unit/test_snn_lang_bridge_stdp.c \
 *       -L build/lib -lnimcp \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_snn_lang_bridge_stdp
 *
 * Coverage:
 *   1. Create a bridge directly (no brain) + register one concept_pop
 *      and one word_pop matching what grounded_language would mirror.
 *   2. Bind once with weight=0.1 to seed a binding.
 *   3. Fire 100 synthesized concept→word spike pairs at increasing
 *      virtual times (mirroring what mirror_binding_to_bridge now does).
 *   4. Read stats: total_ltp_events > 0 and avg_binding_weight > 0.1
 *      (the seeded weight). avg_binding_weight should be close to 0.1
 *      plus the cumulative LTP increments.
 *
 * The bridge's internal STDP path is tested in isolation here — the
 * grounded_language → bridge call chain is verified by integration
 * tests + live training metrics.
 */

#include "snn/bridges/nimcp_snn_language_bridge.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
    exit(1); } } while (0)

int main(void) {
    /* 1. Create bridge with default config. */
    snn_lang_config_t cfg = snn_lang_config_default();
    snn_language_bridge_t* bridge = snn_language_bridge_create(&cfg);
    CHECK(bridge != NULL, "bridge create");

    /* 2. Register one concept + one word. Pop indices below the default
     * capacity so registration can't get rejected. */
    const uint32_t concept_pop = 7;
    const uint32_t word_pop    = 42;
    const uint64_t concept_id  = 0xDEADBEEFCAFEBABEull;
    const char* word_form      = "dog";

    int rc = snn_language_bridge_register_concept(bridge, concept_pop, concept_id);
    CHECK(rc == 0, "register concept");
    rc = snn_language_bridge_register_word(bridge, word_pop, word_form);
    CHECK(rc == 0, "register word");

    /* 3. Seed an initial binding at weight 0.1. */
    rc = snn_language_bridge_bind(bridge, concept_pop, word_pop, 0.1f);
    CHECK(rc == 0, "bind");

    /* Sanity: binding exists with weight ~0.1. */
    snn_lang_stats_t stats0;
    rc = snn_language_bridge_get_stats(bridge, &stats0);
    CHECK(rc == 0, "get_stats baseline");
    CHECK(stats0.active_bindings >= 1, "active_bindings >= 1 after bind");
    CHECK(stats0.total_ltp_events == 0, "no LTP events before spikes");
    /* avg_binding_weight is computed at stats-read time */
    CHECK(stats0.avg_binding_weight > 0.09f && stats0.avg_binding_weight < 0.11f,
          "seed binding weight ~0.1");

    /* 4. Fire 100 concept→word spike pairs spaced 50ms apart, replicating
     * what mirror_binding_to_bridge now does inside grounded_language. */
    float vt = 100.0f;  /* virtual time in ms */
    for (int i = 0; i < 100; i++) {
        rc = snn_language_bridge_concept_spike(bridge, concept_pop, vt);
        CHECK(rc == 0, "concept spike");
        rc = snn_language_bridge_word_spike(bridge, word_pop, vt + 2.0f);
        CHECK(rc == 0, "word spike");
        rc = snn_language_bridge_apply_stdp(bridge, vt + 10.0f);
        CHECK(rc == 0, "apply_stdp");
        vt += 50.0f;
    }

    /* 5. Read back. */
    snn_lang_stats_t stats1;
    rc = snn_language_bridge_get_stats(bridge, &stats1);
    CHECK(rc == 0, "get_stats after spikes");

    fprintf(stderr,
            "[D1] total_ltp_events=%llu total_stdp_updates=%llu "
            "active_bindings=%u avg_binding_weight=%.6f -> %.6f\n",
            (unsigned long long)stats1.total_ltp_events,
            (unsigned long long)stats1.total_stdp_updates,
            stats1.active_bindings,
            stats0.avg_binding_weight,
            stats1.avg_binding_weight);

    /* The whole point: STDP fires ≥ 1 LTP and the seeded binding weight
     * grew. Concept-before-word ordering with Δt=2ms is well inside the
     * 50ms tau_plus window, so every pair should produce LTP.
     *
     * Lower bounds chosen conservatively — STDP machinery may skip pairs
     * whose pre/post traces fell outside the window in this test's
     * timing arithmetic. Anything > 0 is the qualitative win we need
     * (production was stuck at 0). */
    CHECK(stats1.total_ltp_events > 0,
          "total_ltp_events > 0 after 100 concept→word spike pairs");
    CHECK(stats1.avg_binding_weight > 0.1f,
          "avg_binding_weight > 0.1 (initial seed) after LTP");

    /* No LTD should have fired: every pair was concept-before-word. */
    /* (allow any non-zero, but warn loudly if it dominates) */
    fprintf(stderr,
            "[D1] total_ltd_events=%llu (expected ~0)\n",
            (unsigned long long)stats1.total_ltd_events);

    snn_language_bridge_destroy(bridge);
    fprintf(stderr, "[D1] PASS\n");
    return 0;
}
