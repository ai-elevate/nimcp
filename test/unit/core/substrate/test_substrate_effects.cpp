/**
 * @file test_substrate_effects.cpp
 * @brief Unit tests for the shared substrate-effects helper.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

#include "core/substrate/nimcp_substrate_effects.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/memory/nimcp_memory.h"

namespace {

// Build a bare-bones substrate with only the fields the helper reads.
// We don't initialize the mutex; substrate_compute_effects does not lock.
struct SubstrateFixture {
    neural_substrate_t* sub = nullptr;

    SubstrateFixture(float atp, float temp_c, float ion, float mem = 1.0f) {
        sub = static_cast<neural_substrate_t*>(
            nimcp_calloc(1, sizeof(neural_substrate_t)));
        sub->metabolic.atp_level = atp;
        sub->physical.temperature = temp_c;
        sub->physical.ion_balance = ion;
        sub->physical.membrane_integrity = mem;
    }
    ~SubstrateFixture() { nimcp_free(sub); }
};

constexpr float kTol = 1e-4f;

}  // namespace

/* ============================================================================
 * substrate_compute_effects
 * ============================================================================ */

TEST(SubstrateEffects, NullSubstrateReturnsError) {
    axon_substrate_effects_t a;
    dendrite_substrate_effects_t d;
    nimcp_error_t rc = substrate_compute_effects(nullptr, &a, &d);
    EXPECT_EQ(rc, NIMCP_ERROR_NULL_POINTER);
}

TEST(SubstrateEffects, NullOutputsAreSkippedGracefully) {
    SubstrateFixture f(1.0f, 37.0f, 1.0f);
    nimcp_error_t rc = substrate_compute_effects(f.sub, nullptr, nullptr);
    EXPECT_EQ(rc, NIMCP_SUCCESS);
}

TEST(SubstrateEffects, BaselineProducesNoEffectMultipliers) {
    SubstrateFixture f(1.0f, 37.0f, 1.0f, 1.0f);
    axon_substrate_effects_t a{};
    dendrite_substrate_effects_t d{};
    ASSERT_EQ(substrate_compute_effects(f.sub, &a, &d), NIMCP_SUCCESS);

    /* Axon */
    EXPECT_NEAR(a.temperature_q10_factor,   1.0f, kTol);
    EXPECT_NEAR(a.atp_velocity_factor,      1.0f, kTol);
    EXPECT_NEAR(a.myelin_efficiency,        1.0f, kTol);
    EXPECT_NEAR(a.overall_velocity_mod,     1.0f, kTol);
    EXPECT_NEAR(a.ion_gradient_strength,    1.0f, kTol);
    EXPECT_NEAR(a.ap_amplitude_mod,         1.0f, kTol);
    EXPECT_NEAR(a.spike_reliability,        1.0f, kTol);
    EXPECT_NEAR(a.pump_activity,            1.0f, kTol);
    EXPECT_NEAR(a.refractory_period_mod,    1.0f, kTol);
    EXPECT_NEAR(a.transport_efficiency,     1.0f, kTol);
    EXPECT_NEAR(a.membrane_capacitance_mod, 1.0f, kTol);
    EXPECT_NEAR(a.membrane_leak_mod,        1.0f, kTol);
    EXPECT_NEAR(a.overall_capacity,         1.0f, kTol);

    /* Dendrite */
    EXPECT_NEAR(d.membrane_time_constant_mod, 1.0f, kTol);
    EXPECT_NEAR(d.space_constant_mod,         1.0f, kTol);
    EXPECT_NEAR(d.integration_efficiency,     1.0f, kTol);
    EXPECT_NEAR(d.attenuation_mod,            1.0f, kTol);
    EXPECT_NEAR(d.nmda_mg_block_mod,          1.2f, kTol); /* 0.8+0.4×1 */
    EXPECT_NEAR(d.na_channel_availability,    1.0f, kTol);
    EXPECT_NEAR(d.spike_threshold_mod,        1.0f, kTol); /* 2-1 clamped */
    EXPECT_NEAR(d.ca_pump_efficiency,         1.0f, kTol);
    EXPECT_NEAR(d.ca_buffer_capacity,         1.0f, kTol);
    EXPECT_NEAR(d.ca_handling_mod,            1.0f, kTol);
    EXPECT_NEAR(d.ltp_capacity,               1.0f, kTol);
    EXPECT_NEAR(d.plasticity_mod,             1.0f, kTol);
    EXPECT_NEAR(d.overall_capacity,           1.0f, kTol);
}

TEST(SubstrateEffects, HyperthermiaSpeedsKinetics) {
    SubstrateFixture f(1.0f, 47.0f, 1.0f);
    axon_substrate_effects_t a{};
    ASSERT_EQ(substrate_compute_effects(f.sub, &a, nullptr), NIMCP_SUCCESS);
    EXPECT_GT(a.temperature_q10_factor, 1.5f);
}

TEST(SubstrateEffects, HypothermiaSlowsKinetics) {
    SubstrateFixture f(1.0f, 27.0f, 1.0f);
    axon_substrate_effects_t a{};
    ASSERT_EQ(substrate_compute_effects(f.sub, &a, nullptr), NIMCP_SUCCESS);
    EXPECT_LT(a.temperature_q10_factor, 0.5f);
}

TEST(SubstrateEffects, LowATPExtendsRefractoryAndDropsVelocity) {
    SubstrateFixture f(0.5f, 37.0f, 1.0f);
    axon_substrate_effects_t a{};
    ASSERT_EQ(substrate_compute_effects(f.sub, &a, nullptr), NIMCP_SUCCESS);

    EXPECT_NEAR(a.atp_velocity_factor, 0.5f, kTol);
    EXPECT_GT(a.refractory_period_mod, 1.0f);  /* 1/0.5 = 2.0 */
}

/* ============================================================================
 * Inline helpers
 * ============================================================================ */

TEST(SubstrateEffects, ApplyTauScalesByTimeConstantMod) {
    dendrite_substrate_effects_t d{};
    d.membrane_time_constant_mod = 1.5f;
    EXPECT_NEAR(substrate_apply_tau(10.0f, &d), 15.0f, kTol);
}

TEST(SubstrateEffects, ApplyTauWithNullReturnsBase) {
    EXPECT_NEAR(substrate_apply_tau(10.0f, nullptr), 10.0f, kTol);
}

TEST(SubstrateEffects, ApplyTauFloorsSmallValues) {
    dendrite_substrate_effects_t d{};
    d.membrane_time_constant_mod = 0.001f;
    EXPECT_GE(substrate_apply_tau(1.0f, &d), 0.1f);
}

TEST(SubstrateEffects, ApplyTrefScales) {
    axon_substrate_effects_t a{};
    a.refractory_period_mod = 1.5f;
    EXPECT_NEAR(substrate_apply_tref(2.0f, &a), 3.0f, kTol);
    EXPECT_NEAR(substrate_apply_tref(2.0f, nullptr), 2.0f, kTol);
}

TEST(SubstrateEffects, ApplyLrScales) {
    dendrite_substrate_effects_t d{};
    d.plasticity_mod = 0.25f;
    EXPECT_NEAR(substrate_apply_lr(0.04f, &d), 0.01f, kTol);
    EXPECT_NEAR(substrate_apply_lr(0.04f, nullptr), 0.04f, kTol);
}

TEST(SubstrateEffects, SpikeSurvivesBelowReliability) {
    axon_substrate_effects_t a{};
    a.spike_reliability = 0.5f;
    EXPECT_TRUE(substrate_spike_survives(&a, 0.3f));
}

TEST(SubstrateEffects, SpikeFailsAboveReliability) {
    axon_substrate_effects_t a{};
    a.spike_reliability = 0.5f;
    EXPECT_FALSE(substrate_spike_survives(&a, 0.8f));
}

TEST(SubstrateEffects, SpikeSurvivesOnNullEffects) {
    EXPECT_TRUE(substrate_spike_survives(nullptr, 0.0f));
    EXPECT_TRUE(substrate_spike_survives(nullptr, 0.99f));
}

/* ============================================================================
 * substrate_debit_activity
 * ============================================================================ */

TEST(SubstrateEffects, DebitActivityNullSubstrateIsNoOp) {
    substrate_debit_activity(nullptr, 0, 10, 3);  /* must not crash */
    SUCCEED();
}

TEST(SubstrateEffects, DebitActivityDecrementsATP) {
    SubstrateFixture f(1.0f, 37.0f, 1.0f);
    float before = f.sub->metabolic.atp_level;
    substrate_debit_activity(f.sub, 0, 1000, 100);
    float after = f.sub->metabolic.atp_level;
    EXPECT_LT(after, before);
    EXPECT_GE(after, 0.0f);
}
