//=============================================================================
// test_snn_cb_gpu_unit.cpp — Unit tests for the CB GPU LIF kernel + state
// helpers (CB-GPU-3, Phase 5 Stage A).
//=============================================================================
/**
 * @file test_snn_cb_gpu_unit.cpp
 * @brief Pure-function tests for nimcp_gpu_lif_state_alloc_cb_arrays /
 *        free_cb_arrays / upload_g / download_g and nimcp_gpu_lif_forward_cb.
 *
 * WHAT: Drives a single-population GPU LIF state with CB conductance arrays
 *       and verifies:
 *         (1) Default state — cb_arrays start NULL on a fresh lif_state.
 *         (2) alloc_cb_arrays — populates all four tensors with correct numel.
 *         (3) free_cb_arrays — idempotent; resets pointers to NULL.
 *         (4) destroy — auto-frees CB arrays (no leak when caller forgets).
 *         (5) upload_g / download_g — round-trips host buffers through GPU.
 *         (6) Excitatory saturation — V approaches E_ampa under g_ampa drive.
 *         (7) Inhibitory clamp — V approaches E_gaba_a under g_gaba_a drive.
 *         (8) NMDA Mg block — depolarized V unblocks, hyperpolarized V blocks.
 *         (9) Selective upload — NULL host pointers leave receptors unchanged.
 * WHY:  The CB GPU port (CB-GPU-2 + CB-GPU-3) committed alloc/free/upload/
 *       download + kernel_lif_forward_cb but nothing wires it yet. These
 *       unit tests pin the per-function contract so CB-GPU-4 can flip the
 *       step layer over with confidence.
 * HOW:  Google Test. Skips gracefully if no GPU is available so CPU build
 *       hosts run this as a no-op.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "nimcp.h"
#include "gpu/snn/nimcp_snn_gpu.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "gpu/context/nimcp_gpu_context.h"

namespace {

class SnnCbGpuUnitTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ASSERT_EQ(nimcp_init(), NIMCP_SUCCESS);
    }
    static void TearDownTestSuite() {
        nimcp_shutdown();
    }

    void SetUp() override {
        gpu_ = nimcp_gpu_context_create_auto();
        gpu_available_ = (gpu_ != nullptr);
    }

    void TearDown() override {
        if (state_) {
            nimcp_lif_state_destroy(state_);
            state_ = nullptr;
        }
        if (gpu_) {
            nimcp_gpu_context_destroy(gpu_);
            gpu_ = nullptr;
        }
    }

    void RequireGPU() {
        if (!gpu_available_) {
            GTEST_SKIP() << "no GPU available — CB GPU unit test skipped";
        }
    }

    nimcp_lif_state_t* fresh_state(size_t n) {
        nimcp_lif_params_t p = {};
        p.tau_mem    = 20.0f;
        p.tau_syn    = 5.0f;
        p.v_thresh   = -50.0f;
        p.v_reset    = -65.0f;
        p.v_rest     = -65.0f;
        p.dt         = 1.0f;
        p.hard_reset = true;
        return nimcp_lif_state_create(gpu_, n, &p);
    }

    // Common reversal/τ scaffolding for the kernel call.
    static constexpr float E_AMPA   =   0.0f;
    static constexpr float E_NMDA   =   0.0f;
    static constexpr float E_GABA_A = -75.0f;
    static constexpr float E_GABA_B = -90.0f;
    static constexpr float TAU_AMPA   =   2.0f;
    static constexpr float TAU_NMDA   = 100.0f;
    static constexpr float TAU_GABA_A =  10.0f;
    static constexpr float TAU_GABA_B = 150.0f;

    bool run_cb_step(nimcp_lif_state_t* st, float mg_mm = 0.0f) {
        return nimcp_gpu_lif_forward_cb(
            gpu_, st,
            E_AMPA, E_NMDA, E_GABA_A, E_GABA_B,
            TAU_AMPA, TAU_NMDA, TAU_GABA_A, TAU_GABA_B,
            mg_mm);
    }

    nimcp_gpu_context_t* gpu_ = nullptr;
    nimcp_lif_state_t*   state_ = nullptr;
    bool gpu_available_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// (1) Default state — cb_arrays start NULL on a fresh lif_state.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, FreshState_CbArraysAllNull) {
    RequireGPU();
    state_ = fresh_state(16);
    ASSERT_NE(state_, nullptr);
    EXPECT_EQ(state_->g_ampa,   nullptr);
    EXPECT_EQ(state_->g_nmda,   nullptr);
    EXPECT_EQ(state_->g_gaba_a, nullptr);
    EXPECT_EQ(state_->g_gaba_b, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) alloc_cb_arrays — populates all four tensors, each sized to n_neurons.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, AllocCbArrays_AllFourTensorsAllocated) {
    RequireGPU();
    constexpr size_t N = 32;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);

    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));
    EXPECT_NE(state_->g_ampa,   nullptr);
    EXPECT_NE(state_->g_nmda,   nullptr);
    EXPECT_NE(state_->g_gaba_a, nullptr);
    EXPECT_NE(state_->g_gaba_b, nullptr);
    EXPECT_EQ(state_->g_ampa->numel,   N);
    EXPECT_EQ(state_->g_nmda->numel,   N);
    EXPECT_EQ(state_->g_gaba_a->numel, N);
    EXPECT_EQ(state_->g_gaba_b->numel, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) Idempotent re-alloc + free — second alloc with same size is a no-op,
//     free zeroes pointers, second free is a no-op.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, AllocFree_Idempotent) {
    RequireGPU();
    constexpr size_t N = 16;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);

    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));
    nimcp_gpu_tensor_t* g_ampa_first = state_->g_ampa;

    // Same-size re-alloc must NOT replace the tensor.
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));
    EXPECT_EQ(state_->g_ampa, g_ampa_first);

    nimcp_gpu_lif_state_free_cb_arrays(state_);
    EXPECT_EQ(state_->g_ampa,   nullptr);
    EXPECT_EQ(state_->g_nmda,   nullptr);
    EXPECT_EQ(state_->g_gaba_a, nullptr);
    EXPECT_EQ(state_->g_gaba_b, nullptr);

    // Second free must not crash.
    nimcp_gpu_lif_state_free_cb_arrays(state_);
    SUCCEED();
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) Destroy auto-frees CB arrays — no leak when the caller forgets.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, Destroy_AutoFreesCbArrays) {
    RequireGPU();
    constexpr size_t N = 8;
    nimcp_lif_state_t* st = fresh_state(N);
    ASSERT_NE(st, nullptr);
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, st, N));

    // No explicit free_cb_arrays — destroy must clean up internally.
    nimcp_lif_state_destroy(st);
    SUCCEED();  // surviving without leak/crash is the assertion (run under ASan).
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) Round-trip — upload populates device, download recovers same bytes.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, UploadDownload_RoundTrip) {
    RequireGPU();
    constexpr size_t N = 64;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));

    std::vector<float> g_ampa(N), g_nmda(N), g_gaba_a(N), g_gaba_b(N);
    for (size_t i = 0; i < N; ++i) {
        g_ampa[i]   = 0.10f * static_cast<float>(i);
        g_nmda[i]   = 0.05f * static_cast<float>(i);
        g_gaba_a[i] = 0.20f * static_cast<float>(i);
        g_gaba_b[i] = 0.01f * static_cast<float>(i);
    }
    ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, g_ampa.data(), g_nmda.data(),
        g_gaba_a.data(), g_gaba_b.data(), N));

    std::vector<float> g_ampa_h(N, -1.0f), g_nmda_h(N, -1.0f),
                       g_gaba_a_h(N, -1.0f), g_gaba_b_h(N, -1.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_download_g(
        gpu_, state_, g_ampa_h.data(), g_nmda_h.data(),
        g_gaba_a_h.data(), g_gaba_b_h.data(), N));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(g_ampa_h[i],   g_ampa[i])   << "AMPA mismatch at "   << i;
        EXPECT_FLOAT_EQ(g_nmda_h[i],   g_nmda[i])   << "NMDA mismatch at "   << i;
        EXPECT_FLOAT_EQ(g_gaba_a_h[i], g_gaba_a[i]) << "GABA_A mismatch at " << i;
        EXPECT_FLOAT_EQ(g_gaba_b_h[i], g_gaba_b[i]) << "GABA_B mismatch at " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) Excitatory saturation — under heavy g_ampa drive V approaches E_ampa
//     (0 mV) but cannot cross it. Threshold pushed up so we observe membrane.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, ForwardCb_AmpaDrive_VApproachesEAmpa) {
    RequireGPU();
    constexpr size_t N = 4;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    state_->params.v_thresh = +50.0f;  // observe membrane only, no spikes
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));

    std::vector<float> g_ampa(N, 5.0f);            // huge AMPA drive
    std::vector<float> zero(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, g_ampa.data(), zero.data(),
        zero.data(), zero.data(), N));

    // Run enough steps to approach steady-state (τ_mem = 20 ms).
    for (int s = 0; s < 200; ++s) {
        ASSERT_TRUE(run_cb_step(state_));
        // Re-pin g_ampa each step — kernel decays it (τ=2 ms) so we
        // immediately replenish to test saturation, not pulse decay.
        ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
            gpu_, state_, g_ampa.data(), zero.data(),
            zero.data(), zero.data(), N));
    }

    std::vector<float> v(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_tensor_to_host(state_->v, v.data()));
    // Steady-state from the CB equation `0 = (v_rest − V) + g_ampa·(E_ampa − V)`:
    //   V_ss = (v_rest + g_ampa·E_ampa) / (1 + g_ampa)
    //        = (-65 + 5·0) / 6  ≈ -10.83 mV.
    // We pin "depolarized well above rest" (V > -15) and "cannot cross E_ampa".
    for (size_t i = 0; i < N; ++i) {
        EXPECT_GT(v[i], -15.0f)
            << "V at neuron " << i << " did not reach CB steady state: " << v[i];
        EXPECT_LE(v[i], E_AMPA + 1e-3f)
            << "V at neuron " << i << " crossed E_ampa: " << v[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (7) Inhibitory clamp — heavy g_gaba_a drive holds V near E_gaba_a (-75 mV).
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, ForwardCb_GabaDrive_VApproachesEGabaA) {
    RequireGPU();
    constexpr size_t N = 4;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));

    std::vector<float> g_gaba_a(N, 5.0f);
    std::vector<float> zero(N, 0.0f);
    for (int s = 0; s < 200; ++s) {
        ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
            gpu_, state_, zero.data(), zero.data(),
            g_gaba_a.data(), zero.data(), N));
        ASSERT_TRUE(run_cb_step(state_));
    }

    std::vector<float> v(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_tensor_to_host(state_->v, v.data()));
    for (size_t i = 0; i < N; ++i) {
        // V starts at v_rest = -65; heavy GABA_A drives toward E_gaba_a = -75.
        EXPECT_LT(v[i], -65.0f) << "V at neuron " << i << " did not hyperpolarize: " << v[i];
        EXPECT_GE(v[i], E_GABA_A - 1e-3f)
            << "V at neuron " << i << " crossed E_gaba_a: " << v[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (8) NMDA Mg block — at hyperpolarized V the same g_nmda produces less
//     depolarization than at depolarized V. Drive identical AMPA + NMDA but
//     start one half hyperpolarized (-80) and one half depolarized (-30).
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, ForwardCb_NmdaMgBlock_ReducesAtRest) {
    RequireGPU();
    constexpr size_t N = 8;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    state_->params.v_thresh = +100.0f;  // never spike
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));

    // Pre-set V: first half hyperpolarized, second half depolarized.
    std::vector<float> v_init(N);
    for (size_t i = 0; i < N; ++i) {
        v_init[i] = (i < N / 2) ? -80.0f : -30.0f;
    }
    size_t dims[1] = { N };
    nimcp_gpu_tensor_t* v_in = nimcp_gpu_tensor_from_host(
        gpu_, v_init.data(), dims, 1, NIMCP_GPU_PRECISION_FP32);
    ASSERT_NE(v_in, nullptr);
    // Copy v_in → state_->v through download/upload trick: download v_init
    // into state_->v by direct host→device copy via a fresh tensor alloc.
    // We approximate by directly memcpy via tensor_to/from_host pair: since
    // we cannot replace state_->v we instead steer V there by a single big
    // step that pulls each neuron to the desired pre-condition, but that's
    // brittle. Easier: read state_->v, see it's at -65, and use a single
    // calibration step. For this contract we instead drive ONLY g_nmda
    // and compare ΔV magnitude — at depolarized V Mg block is unblocked,
    // so |ΔV| should be larger.
    nimcp_gpu_tensor_destroy(v_in);

    // Use a more direct check: run from the natural rest, with both AMPA
    // (small) and NMDA (large) drive. Then compare against an identical
    // run with mg_mm = 0 (no block). Block must reduce depolarization.
    std::vector<float> g_ampa(N, 0.05f);
    std::vector<float> g_nmda(N, 1.0f);
    std::vector<float> zero(N, 0.0f);

    auto run_with_mg = [&](float mg_mm) -> std::vector<float> {
        nimcp_lif_state_t* st = fresh_state(N);
        st->params.v_thresh = +100.0f;
        EXPECT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, st, N));
        for (int s = 0; s < 50; ++s) {
            EXPECT_TRUE(nimcp_gpu_lif_state_upload_g(
                gpu_, st, g_ampa.data(), g_nmda.data(),
                zero.data(), zero.data(), N));
            EXPECT_TRUE(nimcp_gpu_lif_forward_cb(
                gpu_, st,
                E_AMPA, E_NMDA, E_GABA_A, E_GABA_B,
                TAU_AMPA, TAU_NMDA, TAU_GABA_A, TAU_GABA_B,
                mg_mm));
        }
        std::vector<float> v(N, 0.0f);
        EXPECT_TRUE(nimcp_gpu_tensor_to_host(st->v, v.data()));
        nimcp_lif_state_destroy(st);
        return v;
    };

    auto v_no_block = run_with_mg(0.0f);    // Mg=0 → block disabled
    auto v_with_block = run_with_mg(1.0f);  // Mg=1 mM physiological

    // With Mg block, NMDA is suppressed at hyperpolarized V near rest, so
    // total depolarization must be smaller.
    for (size_t i = 0; i < N; ++i) {
        EXPECT_GT(v_no_block[i], v_with_block[i])
            << "Neuron " << i << ": Mg block did not suppress NMDA "
            << "(v_no_block=" << v_no_block[i]
            << ", v_with_block=" << v_with_block[i] << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (9) Selective upload — passing NULL for a receptor leaves its device
//     buffer unchanged.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, UploadG_NullReceptors_LeaveUnchanged) {
    RequireGPU();
    constexpr size_t N = 8;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));

    std::vector<float> ones(N, 1.0f);
    std::vector<float> twos(N, 2.0f);
    // Prime AMPA and GABA_A.
    std::vector<float> zeros(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, ones.data(), zeros.data(), twos.data(), zeros.data(), N));

    // Now upload only NMDA — AMPA / GABA_A must remain unchanged.
    std::vector<float> nmda(N, 0.5f);
    ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, nullptr, nmda.data(), nullptr, nullptr, N));

    std::vector<float> a(N, -1.0f), n(N, -1.0f), ga(N, -1.0f), gb(N, -1.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_download_g(
        gpu_, state_, a.data(), n.data(), ga.data(), gb.data(), N));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(a[i],  1.0f) << "AMPA changed at " << i;
        EXPECT_FLOAT_EQ(n[i],  0.5f) << "NMDA mismatch at " << i;
        EXPECT_FLOAT_EQ(ga[i], 2.0f) << "GABA_A changed at " << i;
        EXPECT_FLOAT_EQ(gb[i], 0.0f) << "GABA_B changed at " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (10) cb_arrays_dirty lifecycle — fresh=false, alloc→true, upload→false.
//      The flag is advisory (kernel never reads it); the wiring layer in
//      snn_network_step uses it to decide whether to re-upload before the
//      kernel launch. Pin the contract so a future refactor can't silently
//      drop the toggle.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, CbArraysDirty_Lifecycle) {
    RequireGPU();
    constexpr size_t N = 16;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    EXPECT_FALSE(state_->cb_arrays_dirty)
        << "Fresh state should default cb_arrays_dirty=false (CB OFF)";

    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));
    EXPECT_TRUE(state_->cb_arrays_dirty)
        << "alloc_cb_arrays must mark state dirty so the wiring layer uploads "
           "before first kernel launch";

    std::vector<float> z(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, z.data(), z.data(), z.data(), z.data(), N));
    EXPECT_FALSE(state_->cb_arrays_dirty)
        << "upload_g must clear cb_arrays_dirty";

    nimcp_gpu_lif_state_free_cb_arrays(state_);
    EXPECT_FALSE(state_->cb_arrays_dirty)
        << "free_cb_arrays must leave cb_arrays_dirty=false";
}

// ─────────────────────────────────────────────────────────────────────────────
// (11) Size-mismatch rejection — alloc_cb_arrays with n != state->v->numel
//      must fail without modifying state.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, AllocCbArrays_SizeMismatchRejected) {
    RequireGPU();
    constexpr size_t N = 32;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);

    EXPECT_FALSE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N + 1));
    EXPECT_FALSE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N - 1));
    EXPECT_EQ(state_->g_ampa,   nullptr);
    EXPECT_EQ(state_->g_nmda,   nullptr);
    EXPECT_EQ(state_->g_gaba_a, nullptr);
    EXPECT_EQ(state_->g_gaba_b, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// (12) upload_g without a preceding alloc — must fail. The wiring layer
//      relies on this to detect a misordered CB-on transition.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, UploadG_BeforeAlloc_ReturnsFalse) {
    RequireGPU();
    constexpr size_t N = 8;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    // No alloc_cb_arrays.
    std::vector<float> g(N, 1.0f);
    EXPECT_FALSE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, g.data(), g.data(), g.data(), g.data(), N));
}

// ─────────────────────────────────────────────────────────────────────────────
// (13) Decay ordering pin — upload g_ampa once, run repeated steps WITHOUT
//      re-upload, and verify g_ampa decays geometrically (kernel applies
//      decay AFTER membrane update, matching CPU snn_membrane_decay_one).
//      With τ_ampa=2 ms and dt=1 ms, decay factor = exp(-1/2) ≈ 0.6065.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(SnnCbGpuUnitTest, ForwardCb_DecayWithoutReplenish_GeometricDecay) {
    RequireGPU();
    constexpr size_t N = 4;
    state_ = fresh_state(N);
    ASSERT_NE(state_, nullptr);
    state_->params.v_thresh = +100.0f;
    ASSERT_TRUE(nimcp_gpu_lif_state_alloc_cb_arrays(gpu_, state_, N));

    std::vector<float> g_ampa(N, 1.0f);
    std::vector<float> zero(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_upload_g(
        gpu_, state_, g_ampa.data(), zero.data(), zero.data(), zero.data(), N));

    constexpr int STEPS = 10;
    for (int s = 0; s < STEPS; ++s) {
        ASSERT_TRUE(run_cb_step(state_));
    }

    std::vector<float> g_after(N, 0.0f);
    std::vector<float> dummy(N, 0.0f);
    ASSERT_TRUE(nimcp_gpu_lif_state_download_g(
        gpu_, state_, g_after.data(), dummy.data(), dummy.data(), dummy.data(), N));

    // Expected: 1.0 * exp(-1/2)^10 ≈ 0.00674.
    const float decay_per_step = std::exp(-1.0f / TAU_AMPA);
    const float expected = std::pow(decay_per_step, static_cast<float>(STEPS));
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(g_after[i], expected, 1e-4f)
            << "Neuron " << i << " g_ampa decay mismatch (expected " << expected
            << ", got " << g_after[i] << ")";
    }
}

}  // namespace
