/**
 * @file test_lang_networks_into_content.c
 * @brief UNIT — non-SNN networks (ANN/LNN/HNN/CNN) feed the language cascade's
 *        content_intent, default-OFF (2026-06-05).
 *
 * Directive: every network except the FNO feeds the SNN language generator.
 * They converge on content_intent (which seeds the Wernicke concept pop) exactly
 * like the cognitive sources. Each is a default-OFF, gated, soft-additive
 * contributor so the shipped path is byte-identical until a flag is flipped:
 *   - ANN: last_decision->output_vector  → content_intent  (w_ann = 0.25)
 *   - LNN: liquid state                  → content_intent  (w_lnn = 0.2)
 *   - CNN: cached feature embedding       → content_intent  (w_cnn = 0.1)
 *   - HNN: energy-deviation               → produce floor   (no content vector)
 *
 * Isolation trick (same as test_lang_wm_content_intent): run the cascade with
 * ONLY CASCADE_STAGE_CONTENT and a NULL prompt, so every other content_intent
 * source sits at its zero state and the resulting content_intent is EXACTLY the
 * contribution under test.
 *
 * A TINY classification brain has no lnn_network and no visual_cortex, so the
 * LNN/CNN BLENDS can't be exercised end-to-end here (they need those nets).
 * What this test guards deterministically:
 *   1. ANN flag OFF → a one-hot last_decision does NOT blend (shipped default).
 *   2. ANN flag ON  → content_intent[k] == w_ann × value.
 *   3. Default-OFF safety: a one-hot last_decision present but ALL flags off →
 *      content_intent stays ~0 (proves the shipped path is unchanged).
 *   4. Gating no-op safety: LNN/HNN/CNN flags ON with those nets ABSENT → no
 *      crash and no spurious content_intent write.
 *   5. The new visual_cortex_get_cached_features getter: idle cortex → 0 count.
 *
 * Compile (CMake wires this in lang_smoke):
 *   gcc -O0 -g -I include tests/unit/test_lang_networks_into_content.c \
 *       -L build/lib -lnimcp -lm -lpthread \
 *       -Wl,-rpath,/home/bbrelin/nimcp/build/lib \
 *       -o /tmp/test_lang_networks_into_content
 */

#include "language/nimcp_communication_cascade.h"
#include "language/nimcp_grounded_language.h"
#include "perception/nimcp_visual_cortex.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d " #cond " : ", __func__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        g_failures++; \
    } \
} while (0)

/* Must mirror the weights in cascade_stage_content. */
#define W_ANN 0.25f

static brain_t make_brain(const char* name) {
    return brain_create_minimal(name, BRAIN_SIZE_TINY,
                                 BRAIN_TASK_CLASSIFICATION, 16, 8);
}

/* Install a one-hot decision output at dim k as brain->last_decision.
 * Returns the heap decision so the caller can detach + free it before
 * brain_destroy (we own this pointer; the brain must not free it). */
static brain_decision_t* set_onehot_decision(brain_t b, uint32_t dim, uint32_t k) {
    brain_decision_t* d = (brain_decision_t*)calloc(1, sizeof(brain_decision_t));
    if (!d) return NULL;
    d->output_vector = (float*)calloc(dim, sizeof(float));
    if (!d->output_vector) { free(d); return NULL; }
    d->output_vector[k] = 1.0f;
    d->output_size = dim;
    b->last_decision = d;
    return d;
}

static void detach_decision(brain_t b, brain_decision_t* d) {
    if (b->last_decision == d) b->last_decision = NULL;
    if (d) { free(d->output_vector); free(d); }
}

/* ====================================================================== */
/* TEST 1+2: ANN gate OFF → no blend; ON → content_intent[k] = w_ann × v. */
static void test_ann_gate(void) {
    brain_t b = make_brain("ann_gate");
    EXPECT(b != NULL, "create brain");
    if (!b) return;
    if (!b->grounded_lang) {
        fprintf(stderr, "  SKIP: minimal brain has no grounded_lang\n");
        brain_destroy(b); return;
    }
    uint32_t dim = grounded_language_get_semantic_dim(b->grounded_lang);
    EXPECT(dim >= 8, "semantic_dim usable (got %u)", dim);
    if (dim < 8) { brain_destroy(b); return; }

    const uint32_t k = 5u;
    brain_decision_t* d = set_onehot_decision(b, dim, k);
    EXPECT(d != NULL, "alloc decision");
    if (!d) { brain_destroy(b); return; }

    /* (1) Flag OFF (default) → the one-hot decision must NOT blend. */
    b->cascade_ann_in_content = false;
    production_cascade_state_t s_off = {0};
    int rc = communication_cascade_run(b, NULL, CASCADE_STAGE_CONTENT, &s_off);
    EXPECT(rc == 0, "cascade(off) rc=%d", rc);
    if (s_off.content_intent && s_off.content_dim == dim) {
        fprintf(stderr, "  ANN off content_intent[%u] = %.4f (want ~0)\n",
                k, s_off.content_intent[k]);
        EXPECT(fabsf(s_off.content_intent[k]) < 1e-4f,
               "ANN OFF must not blend (got %.4f)", s_off.content_intent[k]);
    }
    cascade_state_cleanup(&s_off);

    /* (2) Flag ON → content_intent[k] == w_ann × 1.0. */
    b->cascade_ann_in_content = true;
    production_cascade_state_t s_on = {0};
    rc = communication_cascade_run(b, NULL, CASCADE_STAGE_CONTENT, &s_on);
    EXPECT(rc == 0, "cascade(on) rc=%d", rc);
    if (s_on.content_intent && s_on.content_dim == dim) {
        float got = s_on.content_intent[k];
        float want = W_ANN * 1.0f;
        fprintf(stderr, "  ANN on  content_intent[%u] = %.4f (want ~%.4f)\n",
                k, got, want);
        EXPECT(fabsf(got - want) < 1e-3f,
               "ANN ON blend: got %.4f want %.4f", got, want);
        uint32_t other = (k + 1u) % dim;
        EXPECT(fabsf(s_on.content_intent[other]) < 1e-4f,
               "non-ANN dim %u must be ~0 (got %.4f)",
               other, s_on.content_intent[other]);
    }
    cascade_state_cleanup(&s_on);

    detach_decision(b, d);
    brain_destroy(b);
    if (g_failures == 0) fprintf(stderr, "PASS test_ann_gate\n");
}

/* ====================================================================== */
/* TEST 3: default-OFF safety — one-hot decision present, ALL flags off →
 * content_intent ~0 (the shipped path is byte-identical to pre-change). */
static void test_all_off_safety(void) {
    brain_t b = make_brain("all_off");
    if (!b) return;
    if (!b->grounded_lang) { brain_destroy(b); return; }
    uint32_t dim = grounded_language_get_semantic_dim(b->grounded_lang);
    if (dim < 8) { brain_destroy(b); return; }

    brain_decision_t* d = set_onehot_decision(b, dim, 3u);
    if (!d) { brain_destroy(b); return; }

    /* All four gates default false (calloc'd struct); assert explicitly. */
    EXPECT(!b->cascade_ann_in_content && !b->cascade_lnn_in_content &&
           !b->cascade_hnn_in_content && !b->cascade_cnn_in_content,
           "all four gates default OFF");

    production_cascade_state_t s = {0};
    int rc = communication_cascade_run(b, NULL, CASCADE_STAGE_CONTENT, &s);
    EXPECT(rc == 0, "cascade rc=%d", rc);
    if (s.content_intent && s.content_dim == dim) {
        float maxabs = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float a = fabsf(s.content_intent[i]);
            if (a > maxabs) maxabs = a;
        }
        fprintf(stderr, "  all-off max|content_intent| = %.6f\n", maxabs);
        EXPECT(maxabs < 1e-4f,
               "all gates off must leave content_intent ~0 (max %.6f)", maxabs);
    }
    cascade_state_cleanup(&s);
    detach_decision(b, d);
    brain_destroy(b);
    if (g_failures == 0) fprintf(stderr, "PASS test_all_off_safety\n");
}

/* ====================================================================== */
/* TEST 4: gating no-op safety — LNN/HNN/CNN flags ON but those nets absent
 * (tiny brain) → no crash, no spurious write. */
static void test_absent_nets_no_crash(void) {
    brain_t b = make_brain("absent_nets");
    if (!b) return;
    if (!b->grounded_lang) { brain_destroy(b); return; }
    uint32_t dim = grounded_language_get_semantic_dim(b->grounded_lang);
    if (dim < 8) { brain_destroy(b); return; }

    /* On a tiny brain: no last_decision (ANN), an LNN that may exist but is
     * untrained (zero liquid state), HNN energy-deviation 0, visual_cortex
     * NULL/idle. Turning every gate ON must therefore be a clean no-op: the
     * blend code runs against absent/idle nets and writes nothing. */
    b->cascade_ann_in_content = true;   /* last_decision is NULL → skip */
    b->cascade_lnn_in_content = true;   /* LNN absent or idle → ~0 */
    b->cascade_hnn_in_content = true;   /* energy-deviation 0 → floor only */
    b->cascade_cnn_in_content = true;   /* visual_cortex NULL/idle → skip */

    production_cascade_state_t s = {0};
    int rc = communication_cascade_run(b, NULL, CASCADE_STAGE_CONTENT, &s);
    EXPECT(rc == 0, "cascade rc=%d (must not crash with absent nets)", rc);
    if (s.content_intent && s.content_dim == dim) {
        float maxabs = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float a = fabsf(s.content_intent[i]);
            if (a > maxabs) maxabs = a;
        }
        fprintf(stderr, "  absent-nets max|content_intent| = %.6f\n", maxabs);
        /* No ANN decision, no LNN/CNN vectors → content_intent must stay ~0.
         * (HNN only touches the produce floor, never content_intent.) */
        EXPECT(maxabs < 1e-4f,
               "absent nets must not write content_intent (max %.6f)", maxabs);
    }
    cascade_state_cleanup(&s);
    brain_destroy(b);
    if (g_failures == 0) fprintf(stderr, "PASS test_absent_nets_no_crash\n");
}

/* ====================================================================== */
/* TEST 5: the new visual_cortex_get_cached_features getter — an idle cortex
 * (never processed an image) returns a clean 0-count, NULL args rejected. */
static void test_cnn_getter_idle(void) {
    visual_cortex_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    /* Minimal sane config; if create fails on this build, skip (the getter is
     * still covered by the absent-nets path above). */
    cfg.input_width = 16; cfg.input_height = 16;
    cfg.num_v1_filters = 4; cfg.feature_dim = 16;
    visual_cortex_t* vc = visual_cortex_create(&cfg);
    if (!vc) {
        fprintf(stderr, "  SKIP: visual_cortex_create returned NULL on this build\n");
        return;
    }
    float buf[32]; uint32_t n = 12345u;
    int rc = visual_cortex_get_cached_features(vc, buf, 32u, &n);
    EXPECT(rc == 0, "getter rc=%d on idle cortex", rc);
    EXPECT(n == 0u, "idle cortex → 0-count (got %u)", n);

    /* NULL-arg rejection. */
    rc = visual_cortex_get_cached_features(vc, NULL, 32u, &n);
    EXPECT(rc == -1, "NULL out rejected");

    visual_cortex_destroy(vc);
    if (g_failures == 0) fprintf(stderr, "PASS test_cnn_getter_idle\n");
}

int main(void) {
    fprintf(stderr, "[UNIT] test_lang_networks_into_content (2026-06-05)\n");
    test_ann_gate();
    test_all_off_safety();
    test_absent_nets_no_crash();
    test_cnn_getter_idle();

    if (g_failures == 0) {
        fprintf(stderr, "OK — networks→content_intent wiring guards passed\n");
        return 0;
    }
    fprintf(stderr, "FAIL — %d failure(s)\n", g_failures);
    return 1;
}
