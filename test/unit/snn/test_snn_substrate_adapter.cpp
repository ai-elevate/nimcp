/**
 * @file test_snn_substrate_adapter.cpp
 * @brief Unit tests for the SNN substrate adapter (Phase 1 biological wiring).
 *
 * WHAT: Covers the four runtime-tunable substrate knobs plus the
 *       snn_network_attach_substrate API. Also runs a small live CSR
 *       network with different substrate states (ATP, temperature) to
 *       verify the effects struct cache is populated and that firing
 *       statistics respond to substrate degradation.
 * WHY:  The substrate adapter is the feedback path from metabolic state
 *       into the hot SNN step. Regressions here silently decouple
 *       biological state from neural dynamics.
 * HOW:  Google Test. Tests that do not need a network just exercise the
 *       tunable API. Tests that need a network construct a lightweight
 *       CSR pop and force the CPU fallback (the GPU fast-path does not
 *       run the substrate hooks yet).
 *
 * @date 2026-04-22
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_types.h"
#include "snn/nimcp_snn_config.h"
#include "snn/nimcp_snn_training.h"
#include "snn/nimcp_snn_synapse.h"
#include "utils/tensor/nimcp_tensor.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "core/substrate/nimcp_substrate_effects.h"
}

/* Internal helper — same pattern the biophysical integration test uses.
 * Declared in nimcp_snn_network.c but not in the public header. */
extern "C" int snn_network_add_population_lightweight(snn_network_t* network,
                                                      uint32_t n_neurons,
                                                      neuron_type_t neuron_type,
                                                      const char* name);

/* Destroy the GPU LIF state so snn_network_step falls back to CPU. */
extern "C" void nimcp_lif_state_destroy(void* state);

/*============================================================================
 * Fixture: saves + restores every substrate/biophysical tunable the
 * tests touch so runs don't leak state into each other.
 *==========================================================================*/
class SNNSubstrateAdapterTest : public ::testing::Test {
protected:
    float saved_enabled           = 0.0f;
    float saved_period            = 0.0f;
    float saved_spike_dropout_on  = 0.0f;
    float saved_plasticity_mod_on = 0.0f;
    /* Biophysics knobs we disable so substrate is the sole modulator. */
    float saved_ahp               = 0.0f;
    float saved_pump              = 0.0f;
    float saved_basket            = 0.0f;
    float saved_noise_rate        = 0.0f;

    void SetUp() override {
        saved_enabled           = snn_tune_get_substrate_enabled();
        saved_period            = snn_tune_get_substrate_update_period();
        saved_spike_dropout_on  = snn_tune_get_substrate_spike_dropout_on();
        saved_plasticity_mod_on = snn_tune_get_substrate_plasticity_mod_on();
        saved_ahp               = snn_tune_get_ahp_enabled();
        saved_pump              = snn_tune_get_pump_enabled();
        saved_basket            = snn_tune_get_basket_enabled();
        saved_noise_rate        = snn_tune_get_noise_rate_hz();
    }

    void TearDown() override {
        snn_tune_set_substrate_enabled(saved_enabled);
        snn_tune_set_substrate_update_period(saved_period);
        snn_tune_set_substrate_spike_dropout_on(saved_spike_dropout_on);
        snn_tune_set_substrate_plasticity_mod_on(saved_plasticity_mod_on);
        snn_tune_set_ahp_enabled(saved_ahp);
        snn_tune_set_pump_enabled(saved_pump);
        snn_tune_set_basket_enabled(saved_basket);
        snn_tune_set_noise_rate_hz(saved_noise_rate);
    }

    /* Build a 1-input / 0-hidden / 1-output scaffold, then add a single
     * lightweight CSR pop of `n_neurons`. Forces CPU fallback so the
     * substrate hooks execute. Returns the lightweight pop pointer (or
     * nullptr on failure). */
    snn_network_t* MakeNetworkWithLightweightPop(uint32_t n_neurons,
                                                 snn_population_t** out_pop) {
        snn_config_t config;
        memset(&config, 0, sizeof(config));
        snn_config_feedforward(&config, 1, 0, 1);
        config.n_populations = 8;  /* room for the added pop */

        snn_network_t* net = snn_network_create(&config);
        snn_config_destroy(&config);
        if (!net) return nullptr;

        /* Force CPU path: drop GPU LIF state if any. */
        if (net->gpu_lif_state) {
            nimcp_lif_state_destroy(net->gpu_lif_state);
            net->gpu_lif_state = nullptr;
        }

        int pid = snn_network_add_population_lightweight(
            net, n_neurons, NEURON_GENERIC_LIF, "substrate_test_pop");
        if (pid < 0) {
            snn_network_destroy(net);
            return nullptr;
        }
        snn_population_t* pop = net->populations[pid];
        if (!pop || !pop->lightweight) {
            snn_network_destroy(net);
            return nullptr;
        }
        /* Empty CSR — no incoming synapses, we drive with external_current. */
        snn_csr_finalize(pop->incoming_csr);

        /* Turn off biophysical mechanisms so substrate is the only modulator. */
        snn_tune_set_ahp_enabled(0.0f);
        snn_tune_set_pump_enabled(0.0f);
        snn_tune_set_basket_enabled(0.0f);
        snn_tune_set_noise_rate_hz(0.0f);

        /* Null threshold_offset to disable intrinsic-plasticity path —
         * it's gated on the pop having all three (threshold_offset,
         * neuron_rate_ema, depression) non-null. We keep depression but
         * null threshold_offset so IP skips. Leak is test-only. */
        pop->threshold_offset = nullptr;

        *out_pop = pop;
        return net;
    }

    static void InjectCurrent(snn_population_t* pop, float amp_mv) {
        ASSERT_NE(pop, nullptr);
        ASSERT_TRUE(pop->lightweight);
        ASSERT_NE(pop->external_current, nullptr);
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            pop->external_current[n] = amp_mv;
        }
    }

    static uint32_t CountSpikes(snn_population_t* pop) {
        const float* s = (const float*)nimcp_tensor_data_const(pop->spike_output);
        if (!s) return 0;
        uint32_t c = 0;
        for (uint32_t n = 0; n < pop->n_neurons; n++) {
            if (s[n] > 0.5f) c++;
        }
        return c;
    }

    /* Reset membrane / refractory / spike_output for a clean rerun. */
    static void HardResetPop(snn_population_t* pop) {
        float* v = (float*)nimcp_tensor_data(pop->membrane_v);
        if (v) for (uint32_t n = 0; n < pop->n_neurons; n++) v[n] = -70.0f;
        float* r = (float*)nimcp_tensor_data(pop->refractory);
        if (r) memset(r, 0, pop->n_neurons * sizeof(float));
        float* s = (float*)nimcp_tensor_data(pop->spike_output);
        if (s) memset(s, 0, pop->n_neurons * sizeof(float));
        if (pop->depression) {
            memset(pop->depression, 0, pop->n_neurons * sizeof(float));
        }
        pop->total_spikes    = 0;
        pop->firing_rate_ema = 0.03f;
        pop->rate_samples    = 0;
    }

    /* Create a fresh substrate with default config, then set the given
     * ATP and temperature. Returns nullptr on failure. */
    neural_substrate_t* MakeSubstrate(float atp, float tempC) {
        substrate_config_t scfg;
        substrate_default_config(&scfg);
        neural_substrate_t* sub = substrate_create(&scfg);
        if (!sub) return nullptr;
        substrate_set_atp(sub, atp);
        substrate_set_temperature(sub, tempC);
        return sub;
    }
};

/*============================================================================
 * Tunables — setters, getters, and out-of-range clamping.
 *==========================================================================*/
TEST_F(SNNSubstrateAdapterTest, TunableEnabledRoundTripsNonzeroToOne) {
    snn_tune_set_substrate_enabled(0.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_enabled(), 0.0f);

    snn_tune_set_substrate_enabled(1.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_enabled(), 1.0f);

    /* Any nonzero -> 1.0 */
    snn_tune_set_substrate_enabled(0.3f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_enabled(), 1.0f);

    snn_tune_set_substrate_enabled(-7.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_enabled(), 1.0f);

    snn_tune_set_substrate_enabled(0.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_enabled(), 0.0f);
}

TEST_F(SNNSubstrateAdapterTest, TunablePeriodClampsOutOfRange) {
    snn_tune_set_substrate_update_period(10.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_update_period(), 10.0f);

    /* Out-of-range low: rejected. */
    snn_tune_set_substrate_update_period(0.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_update_period(), 10.0f);
    snn_tune_set_substrate_update_period(-1.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_update_period(), 10.0f);

    /* Out-of-range high: rejected. */
    snn_tune_set_substrate_update_period(1e9f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_update_period(), 10.0f);

    /* In range: accepted. */
    snn_tune_set_substrate_update_period(5.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_update_period(), 5.0f);

    snn_tune_set_substrate_update_period(10000.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_update_period(), 10000.0f);
}

TEST_F(SNNSubstrateAdapterTest, TunableSpikeDropoutRoundTrip) {
    snn_tune_set_substrate_spike_dropout_on(0.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_spike_dropout_on(), 0.0f);
    snn_tune_set_substrate_spike_dropout_on(1.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_spike_dropout_on(), 1.0f);
    snn_tune_set_substrate_spike_dropout_on(42.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_spike_dropout_on(), 1.0f);
}

TEST_F(SNNSubstrateAdapterTest, TunablePlasticityModRoundTrip) {
    snn_tune_set_substrate_plasticity_mod_on(0.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_plasticity_mod_on(), 0.0f);
    snn_tune_set_substrate_plasticity_mod_on(1.0f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_plasticity_mod_on(), 1.0f);
    snn_tune_set_substrate_plasticity_mod_on(-0.5f);
    EXPECT_FLOAT_EQ(snn_tune_get_substrate_plasticity_mod_on(), 1.0f);
}

/*============================================================================
 * Attach API — null-tolerance + basic state bookkeeping.
 *==========================================================================*/
TEST_F(SNNSubstrateAdapterTest, AttachNullNetworkIsNoOp) {
    /* Should not crash. */
    snn_network_attach_substrate(nullptr, nullptr);
    SUCCEED();
}

TEST_F(SNNSubstrateAdapterTest, AttachSetsAndDetachesPointer) {
    snn_config_t config;
    memset(&config, 0, sizeof(config));
    snn_config_feedforward(&config, 1, 0, 1);
    config.n_populations = 4;
    snn_network_t* net = snn_network_create(&config);
    snn_config_destroy(&config);
    ASSERT_NE(net, nullptr);

    EXPECT_EQ(net->substrate, nullptr);

    substrate_config_t scfg;
    substrate_default_config(&scfg);
    neural_substrate_t* sub = substrate_create(&scfg);
    ASSERT_NE(sub, nullptr);

    snn_network_attach_substrate(net, sub);
    EXPECT_EQ(net->substrate, sub);
    EXPECT_EQ(net->substrate_steps_since_update, 0u);

    /* Detach (NULL) */
    snn_network_attach_substrate(net, nullptr);
    EXPECT_EQ(net->substrate, nullptr);

    substrate_destroy(sub);
    snn_network_destroy(net);
}

/*============================================================================
 * Behavior — network with no substrate runs the baseline path.
 *==========================================================================*/
TEST_F(SNNSubstrateAdapterTest, NullSubstrateBehavesAsBaseline) {
    snn_population_t* pop = nullptr;
    snn_network_t* net = MakeNetworkWithLightweightPop(200, &pop);
    ASSERT_NE(net, nullptr);
    ASSERT_NE(pop, nullptr);

    EXPECT_EQ(net->substrate, nullptr);

    /* With v_reset=-65, v_thresh=-50, tau_mem=20, dt=0.1, and drive=50,
     * analytical time-to-first-spike from v=-70 is ~100 steps (gap 20 mV,
     * decaying at I/tau). Run 200 steps to ensure at least one spike/
     * neuron. */
    snn_tune_set_substrate_enabled(1.0f);
    for (int i = 0; i < 200; i++) {
        InjectCurrent(pop, 50.0f);
        int rc = snn_network_step(net, 0.1f);
        ASSERT_GE(rc, 0);
    }
    EXPECT_GT(pop->total_spikes, 0u);

    snn_network_destroy(net);
}

TEST_F(SNNSubstrateAdapterTest, DisabledKnobIgnoresAttachedSubstrate) {
    snn_population_t* pop = nullptr;
    snn_network_t* net = MakeNetworkWithLightweightPop(200, &pop);
    ASSERT_NE(net, nullptr);

    /* Attach a severely depleted substrate — would normally suppress
     * firing and dim the effects cache. The enabled knob OFF means
     * none of that runs. */
    neural_substrate_t* sub = MakeSubstrate(0.05f /* atp */, 37.0f /* T */);
    ASSERT_NE(sub, nullptr);
    snn_network_attach_substrate(net, sub);

    snn_tune_set_substrate_enabled(0.0f);

    for (int i = 0; i < 200; i++) {
        InjectCurrent(pop, 50.0f);
        int rc = snn_network_step(net, 0.1f);
        ASSERT_GE(rc, 0);
    }

    /* With the knob off the cached effects struct is never filled, so
     * its fields stay at their memset-initialized 0 values. */
    EXPECT_FLOAT_EQ(net->cached_axon_effects.atp_velocity_factor, 0.0f);
    EXPECT_FLOAT_EQ(net->cached_dend_effects.plasticity_mod, 0.0f);

    /* Neurons should still have fired because the substrate was bypassed. */
    EXPECT_GT(pop->total_spikes, 0u);

    snn_network_destroy(net);
    substrate_destroy(sub);
}

/*============================================================================
 * Effects-cache population — attaching + stepping populates the cache.
 *==========================================================================*/
TEST_F(SNNSubstrateAdapterTest, StepPopulatesEffectsCacheFromSubstrate) {
    snn_population_t* pop = nullptr;
    snn_network_t* net = MakeNetworkWithLightweightPop(50, &pop);
    ASSERT_NE(net, nullptr);

    snn_tune_set_substrate_enabled(1.0f);
    snn_tune_set_substrate_update_period(1.0f);  /* refresh every step */

    neural_substrate_t* sub = MakeSubstrate(0.8f, 37.0f);
    ASSERT_NE(sub, nullptr);
    snn_network_attach_substrate(net, sub);

    /* Run one step. The adapter must (1) call substrate_compute_effects,
     * (2) fill the cached struct, (3) not crash. */
    InjectCurrent(pop, 30.0f);
    int rc = snn_network_step(net, 0.1f);
    ASSERT_GE(rc, 0);

    /* The cached struct was populated. atp_velocity_factor is linear in
     * ATP (= atp), so with atp=0.8 the cached value must be near 0.8. */
    EXPECT_NEAR(net->cached_axon_effects.atp_velocity_factor, 0.8f, 1e-4f);
    /* Dendrite plasticity_mod also equals atp in the helper. */
    EXPECT_NEAR(net->cached_dend_effects.plasticity_mod, 0.8f, 1e-4f);
    /* Temperature at 37 °C gives q10_factor = 1.0. */
    EXPECT_NEAR(net->cached_axon_effects.temperature_q10_factor, 1.0f, 1e-4f);

    snn_network_destroy(net);
    substrate_destroy(sub);
}

TEST_F(SNNSubstrateAdapterTest, TemperatureChangesCacheQ10Factor) {
    /* T=45°C raises Q10 factor; T=25°C lowers it. substrate_set_temperature
     * clamps to [20, 45] so we pick values inside that range. The helper
     * is pure arithmetic, so we verify the cache reflects the current
     * temperature after a step. Firing-rate response to temperature is
     * not wired through the current 4-hook surface (tau/tref/spike
     * survival/lr) since the helper's membrane_time_constant_mod
     * depends on membrane integrity, not temperature — but temperature
     * is still cached and available for downstream modules and future
     * wiring. */
    snn_population_t* pop = nullptr;
    snn_network_t* net = MakeNetworkWithLightweightPop(10, &pop);
    ASSERT_NE(net, nullptr);

    snn_tune_set_substrate_enabled(1.0f);
    snn_tune_set_substrate_update_period(1.0f);

    neural_substrate_t* sub = MakeSubstrate(1.0f, 45.0f);
    ASSERT_NE(sub, nullptr);
    snn_network_attach_substrate(net, sub);

    InjectCurrent(pop, 30.0f);
    ASSERT_GE(snn_network_step(net, 0.1f), 0);

    /* Q10 = 2.3^((45-37)/10) = 2.3^0.8 ≈ 1.94 */
    EXPECT_GT(net->cached_axon_effects.temperature_q10_factor, 1.5f);
    float hot_q10 = net->cached_axon_effects.temperature_q10_factor;

    substrate_set_temperature(sub, 25.0f);
    /* Bump counter so the next step recomputes. */
    net->substrate_steps_since_update = 0;
    InjectCurrent(pop, 30.0f);
    ASSERT_GE(snn_network_step(net, 0.1f), 0);

    /* Q10 = 2.3^((25-37)/10) = 2.3^-1.2 ≈ 0.358 */
    EXPECT_LT(net->cached_axon_effects.temperature_q10_factor, 0.6f);
    float cold_q10 = net->cached_axon_effects.temperature_q10_factor;
    EXPECT_LT(cold_q10, hot_q10);

    snn_network_destroy(net);
    substrate_destroy(sub);
}

/*============================================================================
 * ATP-depleted substrate produces FEWER spikes than full ATP over a run.
 * Drive is above threshold so both runs would spike; ATP depletion
 * shrinks spike_reliability (drops spikes) and lengthens the refractory
 * period, giving a detectable reduction.
 *==========================================================================*/
TEST_F(SNNSubstrateAdapterTest, ATPDepletionReducesSpikeCount) {
    /* With v_reset=-65, v_thresh=-50, tau_mem=20, dt=0.1, and a strong
     * constant drive I=50:
     *   - v ramps from -65 toward steady-state -20 with τ=20 ms.
     *   - First-spike latency from v_reset: ~8.1 ms (81 steps).
     *   - Refractory 2 ms (20 steps).
     *   - Then ~81 steps climb to next spike → ~100 steps per spike.
     *
     * We need enough steps that multiple spikes per neuron happen so
     * (a) spike_reliability dropout (atp=0.3 → 0.65) and
     * (b) refractory_period_mod (atp=0.3 → 3.33× longer refractory)
     * both have room to express. 600 steps ≈ 6 spikes/neuron at ATP=1
     * and ≈ 2-3 spikes/neuron at ATP=0.3. */
    snn_population_t* pop = nullptr;
    snn_network_t* net = MakeNetworkWithLightweightPop(300, &pop);
    ASSERT_NE(net, nullptr);

    snn_tune_set_substrate_enabled(1.0f);
    snn_tune_set_substrate_update_period(1.0f);
    snn_tune_set_substrate_spike_dropout_on(1.0f);

    /* Baseline: ATP=1.0 at 37°C. spike_reliability = 1.0,
     * refractory_period_mod = 1.0 — nothing dampens. */
    neural_substrate_t* sub_full = MakeSubstrate(1.0f, 37.0f);
    ASSERT_NE(sub_full, nullptr);
    snn_network_attach_substrate(net, sub_full);

    HardResetPop(pop);
    const int n_steps = 600;
    const float I_drive = 50.0f;
    uint64_t full_atp_spikes = 0;
    for (int i = 0; i < n_steps; i++) {
        InjectCurrent(pop, I_drive);
        snn_network_step(net, 0.1f);
        full_atp_spikes += CountSpikes(pop);
    }

    /* Depleted: ATP=0.3. spike_reliability = 0.65 (35% drop per
     * threshold crossing), refractory_period_mod = 3.33 (t_ref ≈ 6.67
     * ms = 66 steps), plasticity_mod=0.3 (irrelevant here). Combined
     * effect: many fewer spikes over the same window. */
    neural_substrate_t* sub_low = MakeSubstrate(0.3f, 37.0f);
    ASSERT_NE(sub_low, nullptr);
    snn_network_attach_substrate(net, sub_low);

    HardResetPop(pop);
    /* Reset counter so next step recomputes with the new substrate. */
    net->substrate_steps_since_update = 0;

    uint64_t low_atp_spikes = 0;
    for (int i = 0; i < n_steps; i++) {
        InjectCurrent(pop, I_drive);
        snn_network_step(net, 0.1f);
        low_atp_spikes += CountSpikes(pop);
    }

    EXPECT_GT(full_atp_spikes, 0u);
    EXPECT_GT(low_atp_spikes, 0u);
    /* ATP depletion must detectably reduce total spikes. The dominant
     * effect is refractory_period_mod; spike_reliability contributes
     * a few extra drops. Expect at least 15% reduction. */
    EXPECT_LT(low_atp_spikes, (uint64_t)(0.85 * (double)full_atp_spikes))
        << "ATP depletion (0.3 vs 1.0) should reduce firing by >=15% "
        << "(full=" << full_atp_spikes << ", low=" << low_atp_spikes << ")";

    snn_network_destroy(net);
    substrate_destroy(sub_full);
    substrate_destroy(sub_low);
}
