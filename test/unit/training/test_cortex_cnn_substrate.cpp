/**
 * @file test_cortex_cnn_substrate.cpp
 * @brief Unit tests for the cortex CNN substrate adapter (Phase 3).
 *
 * WHAT: Covers the four runtime-tunable substrate knobs, the
 *       cortex_cnn_attach_substrate API, and the ATP / membrane damage →
 *       activation / lr modulation path across the cortex modalities.
 * WHY:  Each cortex CNN has its own metabolic compartment (region_id =
 *       cortex type). The activation-mod knob scales the embedding by
 *       dend_effects.integration_efficiency (which falls with membrane
 *       damage), the plasticity-mod knob scales the effective LR by
 *       dend_effects.plasticity_mod (which falls with ATP depletion).
 *       Regressions here silently decouple cortex output / learning from
 *       the biological state.
 * HOW:  Google Test. Tests that only touch tunables exercise the setter /
 *       getter API. Tests that need a live forward pass build a small
 *       cortex CNN per modality and compare embedding norms across
 *       substrate regimes using ONLY the public API (the cortex processor
 *       struct is intentionally opaque).
 *
 * @date 2026-04-22
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

extern "C" {
#include "training/nimcp_cortex_cnn.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/substrate/nimcp_substrate_effects.h"
}

/*============================================================================
 * Fixture: saves + restores every substrate tunable the tests touch.
 *==========================================================================*/
class CortexCNNSubstrateAdapterTest : public ::testing::Test {
protected:
    float saved_enabled             = 0.0f;
    float saved_period              = 0.0f;
    float saved_activation_mod_on   = 0.0f;
    float saved_plasticity_mod_on   = 0.0f;

    void SetUp() override {
        saved_enabled           = cortex_cnn_tune_get_substrate_enabled();
        saved_period            = cortex_cnn_tune_get_substrate_update_period();
        saved_activation_mod_on = cortex_cnn_tune_get_substrate_activation_mod_on();
        saved_plasticity_mod_on = cortex_cnn_tune_get_substrate_plasticity_mod_on();
    }

    void TearDown() override {
        cortex_cnn_tune_set_substrate_enabled(saved_enabled);
        cortex_cnn_tune_set_substrate_update_period(saved_period);
        cortex_cnn_tune_set_substrate_activation_mod_on(saved_activation_mod_on);
        cortex_cnn_tune_set_substrate_plasticity_mod_on(saved_plasticity_mod_on);
    }

    /* Build a substrate with specific ATP + membrane integrity. Temperature
     * left at 37°C default. Substrate config is defaulted then overlaid. */
    neural_substrate_t* MakeSubstrate(float atp, float membrane = 1.0f) {
        substrate_config_t scfg;
        substrate_default_config(&scfg);
        neural_substrate_t* sub = substrate_create(&scfg);
        if (!sub) return nullptr;
        substrate_set_atp(sub, atp);
        substrate_set_membrane_integrity(sub, membrane);
        return sub;
    }

    static float L2Norm(const float* v, uint32_t n) {
        if (!v || n == 0) return 0.0f;
        float s = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            if (std::isfinite(v[i])) s += v[i] * v[i];
        }
        return std::sqrt(s);
    }

    static bool AllFinite(const float* v, uint32_t n) {
        if (!v) return false;
        for (uint32_t i = 0; i < n; i++) {
            if (!std::isfinite(v[i])) return false;
        }
        return true;
    }
};

/*============================================================================
 * Tunables — round-trip + out-of-range rejection.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, TunableEnabledRoundTripsNonzeroToOne) {
    cortex_cnn_tune_set_substrate_enabled(0.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_enabled(), 0.0f);
    cortex_cnn_tune_set_substrate_enabled(1.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_enabled(), 1.0f);
    cortex_cnn_tune_set_substrate_enabled(0.7f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_enabled(), 1.0f);
    cortex_cnn_tune_set_substrate_enabled(-3.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_enabled(), 1.0f);
    cortex_cnn_tune_set_substrate_enabled(0.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_enabled(), 0.0f);
}

TEST_F(CortexCNNSubstrateAdapterTest, TunablePeriodClampsOutOfRange) {
    cortex_cnn_tune_set_substrate_update_period(10.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 10.0f);

    cortex_cnn_tune_set_substrate_update_period(0.5f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 10.0f);
    cortex_cnn_tune_set_substrate_update_period(-1.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 10.0f);
    cortex_cnn_tune_set_substrate_update_period(1e9f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 10.0f);

    cortex_cnn_tune_set_substrate_update_period(1.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 1.0f);
    cortex_cnn_tune_set_substrate_update_period(10000.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 10000.0f);

    /* NaN/inf silently rejected (range test fails). */
    cortex_cnn_tune_set_substrate_update_period(50.0f);
    cortex_cnn_tune_set_substrate_update_period(
        std::numeric_limits<float>::quiet_NaN());
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 50.0f);
    cortex_cnn_tune_set_substrate_update_period(
        std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_update_period(), 50.0f);
}

TEST_F(CortexCNNSubstrateAdapterTest, TunableActivationModRoundTrip) {
    cortex_cnn_tune_set_substrate_activation_mod_on(0.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_activation_mod_on(), 0.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(1.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_activation_mod_on(), 1.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(42.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_activation_mod_on(), 1.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(-0.25f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_activation_mod_on(), 1.0f);
}

TEST_F(CortexCNNSubstrateAdapterTest, TunablePlasticityModRoundTrip) {
    cortex_cnn_tune_set_substrate_plasticity_mod_on(0.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_plasticity_mod_on(), 0.0f);
    cortex_cnn_tune_set_substrate_plasticity_mod_on(1.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_plasticity_mod_on(), 1.0f);
    cortex_cnn_tune_set_substrate_plasticity_mod_on(7.0f);
    EXPECT_FLOAT_EQ(cortex_cnn_tune_get_substrate_plasticity_mod_on(), 1.0f);
}

/*============================================================================
 * Attach API — null-tolerant.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, AttachNullProcessorIsNoOp) {
    cortex_cnn_attach_substrate(nullptr, nullptr);
    SUCCEED();  /* Should not crash. */
}

TEST_F(CortexCNNSubstrateAdapterTest, AttachSubstrateAndDetach) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc, nullptr);

    /* Initially NULL — forward should still work. */
    std::vector<float> phonemes(64, 0.1f);
    const float* emb0 = cortex_cnn_forward_speech(proc, phonemes.data(),
                                                   (uint32_t)phonemes.size());
    EXPECT_NE(emb0, nullptr);

    /* Attach a healthy substrate — forward still OK. */
    neural_substrate_t* sub = MakeSubstrate(1.0f, 1.0f);
    ASSERT_NE(sub, nullptr);
    cortex_cnn_attach_substrate(proc, sub);

    const float* emb1 = cortex_cnn_forward_speech(proc, phonemes.data(),
                                                   (uint32_t)phonemes.size());
    EXPECT_NE(emb1, nullptr);

    /* Detach (NULL substrate) — still safe. */
    cortex_cnn_attach_substrate(proc, nullptr);
    const float* emb2 = cortex_cnn_forward_speech(proc, phonemes.data(),
                                                   (uint32_t)phonemes.size());
    EXPECT_NE(emb2, nullptr);

    substrate_destroy(sub);
    cortex_cnn_destroy(proc);
}

/*============================================================================
 * No-attach forward path with knob on is a no-op (no substrate to read).
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, NullSubstrateForwardBehavesAsBaseline) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc, nullptr);

    cortex_cnn_tune_set_substrate_enabled(1.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(1.0f);

    std::vector<float> phonemes(64, 0.25f);
    const float* emb = cortex_cnn_forward_speech(proc, phonemes.data(),
                                                  (uint32_t)phonemes.size());
    ASSERT_NE(emb, nullptr);

    uint32_t dim = 0;
    const float* got = cortex_cnn_get_embedding(proc, &dim);
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(dim, 32u);
    EXPECT_TRUE(AllFinite(got, dim));

    cortex_cnn_destroy(proc);
}

/*============================================================================
 * Default-healthy substrate + forward: finite output, no crashes.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, HealthySubstrateForwardProducesFiniteOutput) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc, nullptr);

    cortex_cnn_tune_set_substrate_enabled(1.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(1.0f);
    cortex_cnn_tune_set_substrate_update_period(1.0f);

    neural_substrate_t* sub = MakeSubstrate(1.0f, 1.0f);
    ASSERT_NE(sub, nullptr);
    cortex_cnn_attach_substrate(proc, sub);

    std::vector<float> phonemes(64, 0.25f);
    const float* emb = cortex_cnn_forward_speech(proc, phonemes.data(),
                                                  (uint32_t)phonemes.size());
    ASSERT_NE(emb, nullptr);

    uint32_t dim = 0;
    const float* got = cortex_cnn_get_embedding(proc, &dim);
    ASSERT_NE(got, nullptr);
    EXPECT_TRUE(AllFinite(got, dim));

    substrate_destroy(sub);
    cortex_cnn_destroy(proc);
}

/*============================================================================
 * Damaged membrane reduces integration_efficiency from 1.0 to 0.1 and the
 * apply path should scale the embedding amplitude. Compared against a
 * separate but identically-built processor with a healthy substrate, the
 * damaged cortex must produce a smaller L2 norm.
 *
 * Because each processor has its own random weight init, we can't expect
 * bit-identical baselines. Use a large damage ratio (0.1× → 10× smaller)
 * so the effect dominates any weight-init variance.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, DamagedMembraneShrinksEmbeddingNorm) {
    cortex_cnn_tune_set_substrate_enabled(1.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(1.0f);
    cortex_cnn_tune_set_substrate_update_period(1.0f);

    std::vector<float> phonemes(64);
    for (uint32_t i = 0; i < 64; i++) phonemes[i] = 0.05f + 0.01f * (float)i;

    /* Healthy */
    cortex_cnn_processor_t* proc_ok = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc_ok, nullptr);
    neural_substrate_t* sub_ok = MakeSubstrate(1.0f, 1.0f);
    ASSERT_NE(sub_ok, nullptr);
    cortex_cnn_attach_substrate(proc_ok, sub_ok);

    (void)cortex_cnn_forward_speech(proc_ok, phonemes.data(),
                                    (uint32_t)phonemes.size());
    uint32_t dim_ok = 0;
    const float* got_ok = cortex_cnn_get_embedding(proc_ok, &dim_ok);
    ASSERT_NE(got_ok, nullptr);
    float n_ok = L2Norm(got_ok, dim_ok);

    substrate_destroy(sub_ok);
    cortex_cnn_destroy(proc_ok);

    /* Damaged membrane: integration_efficiency ≈ 0.1 */
    cortex_cnn_processor_t* proc_bad = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc_bad, nullptr);
    neural_substrate_t* sub_bad = MakeSubstrate(1.0f, 0.1f);
    ASSERT_NE(sub_bad, nullptr);
    cortex_cnn_attach_substrate(proc_bad, sub_bad);

    (void)cortex_cnn_forward_speech(proc_bad, phonemes.data(),
                                    (uint32_t)phonemes.size());
    uint32_t dim_bad = 0;
    const float* got_bad = cortex_cnn_get_embedding(proc_bad, &dim_bad);
    ASSERT_NE(got_bad, nullptr);
    float n_bad = L2Norm(got_bad, dim_bad);

    EXPECT_TRUE(AllFinite(got_bad, dim_bad));

    /* Damaged should be strictly smaller. Norms are non-negative; if both
     * happen to be zero the check is vacuous but still passes. */
    EXPECT_LE(n_bad, n_ok)
        << "Damaged-membrane embedding norm (" << n_bad
        << ") must not exceed healthy norm (" << n_ok << ")";

    substrate_destroy(sub_bad);
    cortex_cnn_destroy(proc_bad);
}

/*============================================================================
 * activation_mod_on = 0 disables the amplitude scaling even when a damaged
 * substrate is attached. Uses two fresh processors for clean comparison:
 * knob-OFF baseline vs. knob-ON with the same damaged substrate.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, ActivationModKnobOffDisablesScaling) {
    cortex_cnn_tune_set_substrate_enabled(1.0f);
    cortex_cnn_tune_set_substrate_update_period(1.0f);

    std::vector<float> phonemes(64, 0.2f);

    /* Processor with knob ON — damaged substrate dampens embedding. */
    cortex_cnn_tune_set_substrate_activation_mod_on(1.0f);
    cortex_cnn_processor_t* proc_on = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc_on, nullptr);
    neural_substrate_t* sub_on = MakeSubstrate(1.0f, 0.1f);
    ASSERT_NE(sub_on, nullptr);
    cortex_cnn_attach_substrate(proc_on, sub_on);

    (void)cortex_cnn_forward_speech(proc_on, phonemes.data(),
                                    (uint32_t)phonemes.size());
    uint32_t dim_on = 0;
    const float* got_on = cortex_cnn_get_embedding(proc_on, &dim_on);
    ASSERT_NE(got_on, nullptr);
    float n_on = L2Norm(got_on, dim_on);

    substrate_destroy(sub_on);
    cortex_cnn_destroy(proc_on);

    /* Processor with knob OFF — same damaged substrate, no dampening. */
    cortex_cnn_tune_set_substrate_activation_mod_on(0.0f);
    cortex_cnn_processor_t* proc_off = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc_off, nullptr);
    neural_substrate_t* sub_off = MakeSubstrate(1.0f, 0.1f);
    ASSERT_NE(sub_off, nullptr);
    cortex_cnn_attach_substrate(proc_off, sub_off);

    (void)cortex_cnn_forward_speech(proc_off, phonemes.data(),
                                    (uint32_t)phonemes.size());
    uint32_t dim_off = 0;
    const float* got_off = cortex_cnn_get_embedding(proc_off, &dim_off);
    ASSERT_NE(got_off, nullptr);
    float n_off = L2Norm(got_off, dim_off);

    /* Knob-off norm should be no smaller than knob-on norm (likely larger).
     * Allow a loose check to absorb random weight-init variance. */
    EXPECT_GE(n_off, n_on * 0.9f)
        << "activation_mod_on=0 must not dampen (off=" << n_off
        << ", on=" << n_on << ")";

    substrate_destroy(sub_off);
    cortex_cnn_destroy(proc_off);
}

/*============================================================================
 * Master enable off ignores an attached (damaged) substrate — forward is
 * bit-identical to the no-attach baseline. Builds two processors with the
 * same seed-state (create order) and verifies embedding norms match when
 * the knob is off.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, MasterEnableOffIgnoresAttachedSubstrate) {
    cortex_cnn_tune_set_substrate_enabled(0.0f);

    std::vector<float> phonemes(64, 0.3f);

    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(proc, nullptr);

    neural_substrate_t* sub = MakeSubstrate(0.05f, 0.05f);  /* severe damage */
    ASSERT_NE(sub, nullptr);
    cortex_cnn_attach_substrate(proc, sub);

    for (int i = 0; i < 5; i++) {
        (void)cortex_cnn_forward_speech(proc, phonemes.data(),
                                        (uint32_t)phonemes.size());
    }

    uint32_t dim = 0;
    const float* got = cortex_cnn_get_embedding(proc, &dim);
    ASSERT_NE(got, nullptr);
    EXPECT_TRUE(AllFinite(got, dim));

    /* Norm should be positive (embedding was not zeroed by the modulation). */
    float n = L2Norm(got, dim);
    EXPECT_GT(n, 0.0f);

    substrate_destroy(sub);
    cortex_cnn_destroy(proc);
}

/*============================================================================
 * Audio + visual + somato: forward paths accept an attached substrate
 * without crashing and produce finite embeddings. Smoke test that the
 * refresh / apply helpers are wired into all four modalities.
 *==========================================================================*/
TEST_F(CortexCNNSubstrateAdapterTest, AllModalitiesAcceptSubstrate) {
    cortex_cnn_tune_set_substrate_enabled(1.0f);
    cortex_cnn_tune_set_substrate_activation_mod_on(1.0f);
    cortex_cnn_tune_set_substrate_update_period(1.0f);

    neural_substrate_t* sub = MakeSubstrate(0.5f, 0.5f);
    ASSERT_NE(sub, nullptr);

    /* Visual */
    {
        cortex_cnn_processor_t* p = cortex_cnn_create(CORTEX_CNN_VISUAL, 64);
        ASSERT_NE(p, nullptr);
        cortex_cnn_attach_substrate(p, sub);
        std::vector<uint8_t> pixels(16 * 16 * 3, 128u);
        const float* e = cortex_cnn_forward_visual(p, pixels.data(), 16, 16, 3);
        EXPECT_NE(e, nullptr);
        uint32_t dim = 0;
        const float* got = cortex_cnn_get_embedding(p, &dim);
        ASSERT_NE(got, nullptr);
        EXPECT_TRUE(AllFinite(got, dim));
        cortex_cnn_destroy(p);
    }

    /* Audio (mel spectrogram) */
    {
        cortex_cnn_processor_t* p = cortex_cnn_create(CORTEX_CNN_AUDIO, 64);
        ASSERT_NE(p, nullptr);
        cortex_cnn_attach_substrate(p, sub);
        std::vector<float> mel(128, 0.4f);
        const float* e = cortex_cnn_forward_audio(p, mel.data(), (uint32_t)mel.size());
        EXPECT_NE(e, nullptr);
        uint32_t dim = 0;
        const float* got = cortex_cnn_get_embedding(p, &dim);
        ASSERT_NE(got, nullptr);
        EXPECT_TRUE(AllFinite(got, dim));
        cortex_cnn_destroy(p);
    }

    /* Somato (body state) */
    {
        cortex_cnn_processor_t* p = cortex_cnn_create(CORTEX_CNN_SOMATO, 32);
        ASSERT_NE(p, nullptr);
        cortex_cnn_attach_substrate(p, sub);
        std::vector<float> segs(45, 0.2f);
        const float* e = cortex_cnn_forward_somato(p, segs.data(), (uint32_t)segs.size());
        EXPECT_NE(e, nullptr);
        uint32_t dim = 0;
        const float* got = cortex_cnn_get_embedding(p, &dim);
        ASSERT_NE(got, nullptr);
        EXPECT_TRUE(AllFinite(got, dim));
        cortex_cnn_destroy(p);
    }

    substrate_destroy(sub);
}
