//=============================================================================
// test_snn_membrane_unit.cpp — Unit tests for CB membrane helpers
//=============================================================================
/**
 * @file test_snn_membrane_unit.cpp
 * @brief Pure-function tests for nimcp_snn_membrane.h.
 *
 * WHAT: Validates the three header-inline helpers added by the CB migration
 *       — snn_membrane_compute_dv, snn_membrane_decay_one,
 *       snn_membrane_deposit_synapse — in isolation, without booting any
 *       SNN populations or brain init.
 * WHY:  These helpers are the single source of truth for the membrane
 *       equation; if they are wrong, every CB-mode population is wrong.
 *       Pure functions are trivially unit-testable, and bit-identical
 *       OFF-mode behavior must be guaranteed before integration tests.
 * HOW:  Direct calls into the static-inline helpers from a C++ file.
 */

#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "snn/nimcp_snn_membrane.h"
}

//=============================================================================
// snn_membrane_compute_dv — current mode (OFF, must be bit-identical)
//=============================================================================

TEST(SnnMembraneComputeDv, CurrentMode_RestEquilibrium_DvZero) {
    // At rest with no input, dv must be exactly zero.
    float dv = snn_membrane_compute_dv(
        /*v=*/-65.0f, /*v_rest=*/-65.0f,
        /*tau=*/20.0f, /*dt=*/1.0f,
        /*i_syn=*/0.0f,
        /*g_exc=*/0.0f, /*g_inh=*/0.0f,
        /*e_exc=*/0.0f, /*e_inh=*/-80.0f,
        /*cb=*/false);
    EXPECT_FLOAT_EQ(dv, 0.0f);
}

TEST(SnnMembraneComputeDv, CurrentMode_BelowRest_PullsUp) {
    // V below rest with no synaptic drive → leak pulls V up toward rest.
    float dv = snn_membrane_compute_dv(
        -75.0f, -65.0f, 20.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f, -80.0f, false);
    // dv = (v_rest - v + I_syn) / tau * dt = (10) / 20 * 1 = 0.5
    EXPECT_NEAR(dv, 0.5f, 1e-5f);
}

TEST(SnnMembraneComputeDv, CurrentMode_BitIdenticalToLegacyEquation) {
    // Sweep representative (v, I_syn) values and verify dv exactly matches
    // (v_rest - v + I_syn) / tau * dt — the legacy hot-loop expression.
    const float v_rest = -65.0f, tau = 20.0f, dt = 1.0f;
    for (float v = -80; v <= -40; v += 5) {
        for (float i_syn = -20; i_syn <= 20; i_syn += 5) {
            float expected = (v_rest - v + i_syn) / tau * dt;
            float dv = snn_membrane_compute_dv(
                v, v_rest, tau, dt, i_syn,
                0.0f, 0.0f, 0.0f, -80.0f, false);
            // The 100mV clamp should NOT trip in this range — verify.
            EXPECT_FLOAT_EQ(dv, expected)
                << "Mismatch at v=" << v << " I_syn=" << i_syn;
        }
    }
}

//=============================================================================
// snn_membrane_compute_dv — conductance mode
//=============================================================================

TEST(SnnMembraneComputeDv, CbMode_RestNoConductance_DvZero) {
    // V = v_rest, g_exc = g_inh = 0 → dv = 0.
    float dv = snn_membrane_compute_dv(
        -65.0f, -65.0f, 20.0f, 1.0f,
        /*i_syn=*/123.0f /* must be ignored in CB */,
        0.0f, 0.0f, 0.0f, -80.0f, true);
    EXPECT_FLOAT_EQ(dv, 0.0f);
}

TEST(SnnMembraneComputeDv, CbMode_ExcitatoryConductance_DepolarizesTowardEexc) {
    // V at rest, large g_exc, no g_inh: dv must be positive (depolarizing)
    // and proportional to (E_exc - V).
    float v = -65.0f, e_exc = 0.0f;
    float g_exc = 0.5f;
    float dv = snn_membrane_compute_dv(
        v, -65.0f, 20.0f, 1.0f, 0.0f,
        g_exc, 0.0f, e_exc, -80.0f, true);
    // dv = (0 + g_exc*(E_exc - v) + 0) / tau * dt = (0.5 * 65) / 20 = 1.625
    EXPECT_NEAR(dv, 1.625f, 1e-5f);
    EXPECT_GT(dv, 0.0f);  // depolarizing
}

TEST(SnnMembraneComputeDv, CbMode_InhibitoryConductance_HyperpolarizesTowardEinh) {
    // V at rest, large g_inh: dv negative (V > E_inh, so drive points down).
    float v = -65.0f, e_inh = -80.0f;
    float g_inh = 0.3f;
    float dv = snn_membrane_compute_dv(
        v, -65.0f, 20.0f, 1.0f, 0.0f,
        0.0f, g_inh, 0.0f, e_inh, true);
    // dv = (0 + 0 + g_inh*(E_inh - v)) / tau * dt = (0.3 * -15) / 20 = -0.225
    EXPECT_NEAR(dv, -0.225f, 1e-5f);
    EXPECT_LT(dv, 0.0f);
}

TEST(SnnMembraneComputeDv, CbMode_NeverOvershootsEexc) {
    // The key CB property: V never exceeds E_exc, no matter how strong
    // g_exc grows. (V approaches a leak/conductance equilibrium below
    // E_exc — leak pulls it back. With g_exc → ∞ the equilibrium → E_exc.)
    float v = -65.0f;
    const float g_exc = 1.0f;
    for (int step = 0; step < 1000; step++) {
        float dv = snn_membrane_compute_dv(
            v, -65.0f, 20.0f, 1.0f, 0.0f,
            g_exc, 0.0f, 0.0f, -80.0f, true);
        v += dv;
        EXPECT_LE(v, 0.05f) << "V exceeded E_exc at step " << step;
    }
    // Verify equilibrium is in the analytically-correct place:
    // 0 = (v_rest - v) + g_exc*(E_exc - v) → v_eq = (v_rest + g_exc*E_exc)/(1+g_exc)
    // = (-65 + 0) / 2 = -32.5 mV.
    EXPECT_NEAR(v, -32.5f, 0.5f);
}

TEST(SnnMembraneComputeDv, CbMode_StrongerGexcKeepsVNearEexc) {
    // Larger g_exc should push V much closer to E_exc than weak g_exc.
    // With forward Euler at dt=1ms and tau=20ms, very large g_exc creates
    // a stiff system that oscillates a few mV around the analytic
    // equilibrium — that's a discretization artifact, not a CB bug.
    // The invariant we care about: V never exceeds E_exc, and V stays in
    // a tight band around it (much tighter than the g_exc=1 case).
    float v = -65.0f;
    const float g_exc = 100.0f;
    float v_min_after_settling = 1e9f;
    for (int step = 0; step < 1000; step++) {
        float dv = snn_membrane_compute_dv(
            v, -65.0f, 20.0f, 1.0f, 0.0f,
            g_exc, 0.0f, 0.0f, -80.0f, true);
        v += dv;
        EXPECT_LE(v, 0.05f) << "V exceeded E_exc at step " << step;
        if (step >= 100 && v < v_min_after_settling) v_min_after_settling = v;
    }
    // Once settled, V stays within ~5 mV of E_exc (vs 32.5 mV gap with g_exc=1).
    EXPECT_GE(v_min_after_settling, -5.0f)
        << "V drifted further than 5 mV from E_exc with g_exc=100";
}

TEST(SnnMembraneComputeDv, CbMode_NeverUndershootsEinh) {
    // Symmetric: V never goes below E_inh.
    float v = -65.0f;
    const float g_inh = 1.0f;
    for (int step = 0; step < 1000; step++) {
        float dv = snn_membrane_compute_dv(
            v, -65.0f, 20.0f, 1.0f, 0.0f,
            0.0f, g_inh, 0.0f, -80.0f, true);
        v += dv;
        EXPECT_GE(v, -80.05f) << "V undershot E_inh at step " << step;
    }
    // v_eq = (-65 + 1*-80) / 2 = -72.5 mV (analytical).
    EXPECT_NEAR(v, -72.5f, 0.5f);
}

//=============================================================================
// dv clamp (biophysical guard)
//=============================================================================

TEST(SnnMembraneComputeDv, DvClamp_PositiveOverflowClampedTo100) {
    // tau very small + huge drive → dv would explode without the clamp.
    float dv = snn_membrane_compute_dv(
        -65.0f, -65.0f, /*tau=*/0.001f, /*dt=*/1.0f,
        /*i_syn=*/10000.0f,
        0.0f, 0.0f, 0.0f, -80.0f, false);
    EXPECT_LE(dv, 100.0f);
    EXPECT_FLOAT_EQ(dv, 100.0f);
}

TEST(SnnMembraneComputeDv, DvClamp_NegativeOverflowClampedToNeg100) {
    float dv = snn_membrane_compute_dv(
        -65.0f, -65.0f, 0.001f, 1.0f,
        /*i_syn=*/-10000.0f,
        0.0f, 0.0f, 0.0f, -80.0f, false);
    EXPECT_GE(dv, -100.0f);
    EXPECT_FLOAT_EQ(dv, -100.0f);
}

TEST(SnnMembraneComputeDv, DvClamp_NormalRangeUnaffected) {
    // Sane inputs must NOT trip the clamp.
    float dv = snn_membrane_compute_dv(
        -65.0f, -65.0f, 20.0f, 1.0f,
        /*i_syn=*/15.0f,
        0.0f, 0.0f, 0.0f, -80.0f, false);
    EXPECT_LT(dv, 5.0f);
    EXPECT_GT(dv, 0.0f);
    EXPECT_FLOAT_EQ(dv, 15.0f / 20.0f);
}

TEST(SnnMembraneComputeDv, TauZeroFloored_DoesNotCrashOrReturnNaN) {
    // Pathological tau=0 should be floored to 1e-3 silently.
    float dv = snn_membrane_compute_dv(
        -65.0f, -65.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 0.0f, -80.0f, false);
    EXPECT_TRUE(std::isfinite(dv));
}

//=============================================================================
// snn_membrane_decay_one
//=============================================================================

TEST(SnnMembraneDecayOne, ScalesByDecayFactor) {
    float g_exc = 1.0f, g_inh = 0.5f;
    snn_membrane_decay_one(&g_exc, &g_inh, 0.5f, 0.25f);
    EXPECT_FLOAT_EQ(g_exc, 0.5f);
    EXPECT_FLOAT_EQ(g_inh, 0.125f);
}

TEST(SnnMembraneDecayOne, RepeatedDecayConvergesToZero) {
    float g = 1.0f, g_unused = 0.0f;
    const float decay = 0.9f;
    for (int i = 0; i < 100; i++) {
        snn_membrane_decay_one(&g, &g_unused, decay, decay);
    }
    EXPECT_LT(g, 0.001f);  // 0.9^100 ≈ 2.66e-5
    EXPECT_FLOAT_EQ(g_unused, 0.0f);
}

TEST(SnnMembraneDecayOne, NullPointersAreNoOp) {
    // NULL inputs must not crash (silent-degrade contract).
    snn_membrane_decay_one(nullptr, nullptr, 0.5f, 0.5f);
    SUCCEED();  // no crash = pass
}

TEST(SnnMembraneDecayOne, OnlyExcSet_OnlyExcDecays) {
    float g_exc = 1.0f;
    snn_membrane_decay_one(&g_exc, nullptr, 0.5f, 0.5f);
    EXPECT_FLOAT_EQ(g_exc, 0.5f);
}

//=============================================================================
// snn_membrane_deposit_synapse
//=============================================================================

TEST(SnnMembraneDeposit, CurrentMode_PositiveWeight_AddsToISyn) {
    float i_syn = 0.0f, g_exc = 0.0f, g_inh = 0.0f;
    snn_membrane_deposit_synapse(&i_syn, &g_exc, &g_inh, 5.0f, /*cb=*/false);
    EXPECT_FLOAT_EQ(i_syn, 5.0f);
    EXPECT_FLOAT_EQ(g_exc, 0.0f);
    EXPECT_FLOAT_EQ(g_inh, 0.0f);
}

TEST(SnnMembraneDeposit, CurrentMode_NegativeWeight_AddsToISyn) {
    float i_syn = 10.0f, g_exc = 0.0f, g_inh = 0.0f;
    snn_membrane_deposit_synapse(&i_syn, &g_exc, &g_inh, -3.0f, false);
    EXPECT_FLOAT_EQ(i_syn, 7.0f);  // sign-preserving sum
    EXPECT_FLOAT_EQ(g_exc, 0.0f);
    EXPECT_FLOAT_EQ(g_inh, 0.0f);
}

TEST(SnnMembraneDeposit, CbMode_PositiveWeight_RoutesToGexc) {
    float i_syn = 0.0f, g_exc = 0.1f, g_inh = 0.2f;
    snn_membrane_deposit_synapse(&i_syn, &g_exc, &g_inh, 0.5f, /*cb=*/true);
    EXPECT_FLOAT_EQ(i_syn, 0.0f);    // I_syn untouched in CB mode
    EXPECT_FLOAT_EQ(g_exc, 0.6f);    // 0.1 + 0.5
    EXPECT_FLOAT_EQ(g_inh, 0.2f);    // unchanged
}

TEST(SnnMembraneDeposit, CbMode_NegativeWeight_RoutesAbsToGinh) {
    float i_syn = 0.0f, g_exc = 0.1f, g_inh = 0.2f;
    snn_membrane_deposit_synapse(&i_syn, &g_exc, &g_inh, -0.5f, true);
    EXPECT_FLOAT_EQ(i_syn, 0.0f);
    EXPECT_FLOAT_EQ(g_exc, 0.1f);
    EXPECT_FLOAT_EQ(g_inh, 0.7f);    // 0.2 + |-0.5|
}

TEST(SnnMembraneDeposit, CbMode_ZeroWeight_RoutesToGexc_NoChange) {
    // Zero is non-negative → routes to g_exc with delta zero.
    float i_syn = 0.0f, g_exc = 0.5f, g_inh = 0.5f;
    snn_membrane_deposit_synapse(&i_syn, &g_exc, &g_inh, 0.0f, true);
    EXPECT_FLOAT_EQ(g_exc, 0.5f);
    EXPECT_FLOAT_EQ(g_inh, 0.5f);
}

TEST(SnnMembraneDeposit, CbMode_NullGexc_PositiveWeightDropped) {
    // Silent-degrade contract: missing conductance arrays mean no-op for
    // their direction. No crash, no I_syn write.
    float i_syn = 0.0f, g_inh = 0.0f;
    snn_membrane_deposit_synapse(&i_syn, /*g_exc=*/nullptr, &g_inh, 1.0f, true);
    EXPECT_FLOAT_EQ(i_syn, 0.0f);
    EXPECT_FLOAT_EQ(g_inh, 0.0f);
}

TEST(SnnMembraneDeposit, CurrentMode_NullISyn_NoOpNoCrash) {
    snn_membrane_deposit_synapse(nullptr, nullptr, nullptr, 5.0f, false);
    SUCCEED();
}

//=============================================================================
// Composition: deposit + decay + integrate over multiple steps
//=============================================================================

TEST(SnnMembraneComposition, CbMode_SingleSpikeDecaysWithTau) {
    // Inject one spike (deposit weight=1.0 to g_exc), then decay over many
    // steps with tau_exc=2ms, dt=1ms. After 5 tau ≈ 10 steps, g_exc < 1%.
    float g_exc = 0.0f, g_inh = 0.0f;
    snn_membrane_deposit_synapse(/*i_syn=*/nullptr, &g_exc, &g_inh, 1.0f, true);
    EXPECT_FLOAT_EQ(g_exc, 1.0f);

    const float decay = std::exp(-1.0f / 2.0f);  // dt/tau = 0.5
    for (int step = 0; step < 10; step++) {
        snn_membrane_decay_one(&g_exc, &g_inh, decay, decay);
    }
    EXPECT_LT(g_exc, 0.01f) << "g_exc did not decay to <1% in 10 steps";
}

TEST(SnnMembraneComposition, CbMode_BalancedEandI_VStaysAtRest) {
    // E and I deposits that exactly cancel should leave V at rest.
    // g_exc * (E_exc - V_rest) = g_inh * (V_rest - E_inh)
    // g_exc * 65 = g_inh * 15  →  g_inh = g_exc * 65/15
    float v = -65.0f, g_exc = 0.1f, g_inh = 0.1f * (65.0f / 15.0f);
    for (int step = 0; step < 100; step++) {
        float dv = snn_membrane_compute_dv(
            v, -65.0f, 20.0f, 1.0f, 0.0f,
            g_exc, g_inh, 0.0f, -80.0f, true);
        v += dv;
    }
    EXPECT_NEAR(v, -65.0f, 0.5f);
}
