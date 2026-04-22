/**
 * @file test_snn_substrate_integration.cpp
 * @brief Integration tests for the SNN substrate adapter in realistic
 *        multi-step runs alongside Wave A+B biophysics.
 *
 * WHAT: Exercises snn_network_attach_substrate + snn_network_step with
 *       the Wave A+B mechanisms (AHP/pump/basket/E-I noise) ALL enabled,
 *       verifying (a) ATP ramp-down reduces firing, (b) no NaN/Inf or
 *       crash when every biophysics + substrate knob is on, and
 *       (c) the cache refresh counter tracks the configured period.
 * WHY:  The substrate must compose with existing biophysics without
 *       destabilizing dynamics; isolated unit tests don't catch
 *       interaction regressions.
 * HOW:  Google Test. Uses the same lightweight-CSR + CPU-fallback setup
 *       as the biophysical integration test.
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

extern "C" int snn_network_add_population_lightweight(snn_network_t* network,
                                                      uint32_t n_neurons,
                                                      neuron_type_t neuron_type,
                                                      const char* name);
extern "C" void nimcp_lif_state_destroy(void* state);

/*============================================================================
 * Fixture. Saves/restores every tunable we touch.
 *==========================================================================*/
class SNNSubstrateIntegrationTest : public ::testing::Test {
protected:
    snn_network_t* network  = nullptr;
    neural_substrate_t* sub = nullptr;

    float saved_enabled           = 0.0f;
    float saved_period            = 0.0f;
    float saved_dropout           = 0.0f;
    float saved_plast_on          = 0.0f;
    float saved_ahp               = 0.0f;
    float saved_pump              = 0.0f;
    float saved_basket            = 0.0f;
    float saved_noise_rate        = 0.0f;

    void SetUp() override {
        saved_enabled    = snn_tune_get_substrate_enabled();
        saved_period     = snn_tune_get_substrate_update_period();
        saved_dropout    = snn_tune_get_substrate_spike_dropout_on();
        saved_plast_on   = snn_tune_get_substrate_plasticity_mod_on();
        saved_ahp        = snn_tune_get_ahp_enabled();
        saved_pump       = snn_tune_get_pump_enabled();
        saved_basket     = snn_tune_get_basket_enabled();
        saved_noise_rate = snn_tune_get_noise_rate_hz();
    }

    void TearDown() override {
        if (network) {
            snn_network_destroy(network);
            network = nullptr;
        }
        if (sub) {
            substrate_destroy(sub);
            sub = nullptr;
        }
        snn_tune_set_substrate_enabled(saved_enabled);
        snn_tune_set_substrate_update_period(saved_period);
        snn_tune_set_substrate_spike_dropout_on(saved_dropout);
        snn_tune_set_substrate_plasticity_mod_on(saved_plast_on);
        snn_tune_set_ahp_enabled(saved_ahp);
        snn_tune_set_pump_enabled(saved_pump);
        snn_tune_set_basket_enabled(saved_basket);
        snn_tune_set_noise_rate_hz(saved_noise_rate);
    }

    snn_population_t* BuildLightweight(uint32_t n_neurons) {
        snn_config_t config;
        memset(&config, 0, sizeof(config));
        snn_config_feedforward(&config, 1, 0, 1);
        config.n_populations = 8;

        network = snn_network_create(&config);
        snn_config_destroy(&config);
        EXPECT_NE(network, nullptr);
        if (!network) return nullptr;
        if (network->gpu_lif_state) {
            nimcp_lif_state_destroy(network->gpu_lif_state);
            network->gpu_lif_state = nullptr;
        }

        int pid = snn_network_add_population_lightweight(
            network, n_neurons, NEURON_GENERIC_LIF, "integration_pop");
        EXPECT_GE(pid, 0);
        if (pid < 0) return nullptr;

        snn_population_t* pop = network->populations[pid];
        EXPECT_NE(pop, nullptr);
        EXPECT_TRUE(pop && pop->lightweight);
        snn_csr_finalize(pop->incoming_csr);

        /* Disable IP so we're measuring biophysics + substrate, not
         * threshold-offset drift. Leak is test-only. */
        if (pop) pop->threshold_offset = nullptr;
        return pop;
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
        for (uint32_t n = 0; n < pop->n_neurons; n++) if (s[n] > 0.5f) c++;
        return c;
    }
    static void HardReset(snn_population_t* pop) {
        float* v = (float*)nimcp_tensor_data(pop->membrane_v);
        if (v) for (uint32_t n = 0; n < pop->n_neurons; n++) v[n] = -70.0f;
        float* r = (float*)nimcp_tensor_data(pop->refractory);
        if (r) memset(r, 0, pop->n_neurons * sizeof(float));
        float* s = (float*)nimcp_tensor_data(pop->spike_output);
        if (s) memset(s, 0, pop->n_neurons * sizeof(float));
        if (pop->depression) memset(pop->depression, 0, pop->n_neurons * sizeof(float));
        pop->total_spikes = 0;
        pop->firing_rate_ema = 0.03f;
        pop->rate_samples = 0;
    }
};

/*============================================================================
 * Test 1: ATP ramp — firing goes DOWN as ATP runs out over a 200-step run.
 * Biophysics all off so substrate is the dominant modulator. Compare first
 * vs last 50-step windows within a single continuous run.
 *==========================================================================*/
TEST_F(SNNSubstrateIntegrationTest, ATPRampDownReducesFiringOverTime) {
    /* Biophysics off — we want the substrate alone to drive the ramp
     * effect. */
    snn_tune_set_ahp_enabled(0.0f);
    snn_tune_set_pump_enabled(0.0f);
    snn_tune_set_basket_enabled(0.0f);
    snn_tune_set_noise_rate_hz(0.0f);

    snn_population_t* pop = BuildLightweight(500);
    ASSERT_NE(pop, nullptr);

    /* Substrate knobs: everything on, fresh every step so the ATP ramp
     * is felt immediately. */
    snn_tune_set_substrate_enabled(1.0f);
    snn_tune_set_substrate_update_period(1.0f);
    snn_tune_set_substrate_spike_dropout_on(1.0f);
    snn_tune_set_substrate_plasticity_mod_on(1.0f);

    substrate_config_t scfg;
    substrate_default_config(&scfg);
    sub = substrate_create(&scfg);
    ASSERT_NE(sub, nullptr);
    substrate_set_atp(sub, 1.0f);
    substrate_set_temperature(sub, 37.0f);
    snn_network_attach_substrate(network, sub);

    HardReset(pop);

    /* 2000 steps with first/last 500-step windows. First-spike latency
     * is ~100 steps from v_rest so a 500-step window catches 4-5 firing
     * cycles per neuron; enough to average out stochastic variation in
     * the spike-reliability dropout. Compare first window (ATP full) to
     * last window (ATP depleted). */
    const int n_steps    = 2000;
    const int window     = 500;
    const float I_drive  = 55.0f;
    uint64_t first_window_spikes = 0;
    uint64_t last_window_spikes  = 0;

    for (int step = 0; step < n_steps; step++) {
        /* Linear ATP ramp 1.0 -> 0.3 over the full run. */
        float alpha = (float)step / (float)(n_steps - 1);
        float atp   = 1.0f - 0.7f * alpha;  /* 1.0 -> 0.3 */
        substrate_set_atp(sub, atp);

        InjectCurrent(pop, I_drive);
        snn_network_step(network, 0.1f);
        uint32_t s = CountSpikes(pop);
        if (step < window)        first_window_spikes += s;
        if (step >= n_steps - window) last_window_spikes += s;
    }

    EXPECT_GT(first_window_spikes, 0u);
    EXPECT_LT(last_window_spikes, first_window_spikes)
        << "ATP ramp 1.0 -> 0.3 should reduce firing in the last window "
        << "(first=" << first_window_spikes
        << ", last=" << last_window_spikes << ")";
}

/*============================================================================
 * Test 2: Biophysics + substrate composition — full stack on, run must be
 * stable (no NaN/Inf, at least some spikes, no step error).
 *==========================================================================*/
TEST_F(SNNSubstrateIntegrationTest, AllWaveABEnabledPlusSubstrateStaysStable) {
    /* Everything on, defaults. */
    snn_tune_set_ahp_enabled(1.0f);
    snn_tune_set_pump_enabled(1.0f);
    snn_tune_set_basket_enabled(1.0f);
    snn_tune_set_noise_rate_hz(20.0f);

    snn_tune_set_substrate_enabled(1.0f);
    snn_tune_set_substrate_update_period(10.0f);
    snn_tune_set_substrate_spike_dropout_on(1.0f);
    snn_tune_set_substrate_plasticity_mod_on(1.0f);

    snn_population_t* pop = BuildLightweight(200);
    ASSERT_NE(pop, nullptr);

    substrate_config_t scfg;
    substrate_default_config(&scfg);
    sub = substrate_create(&scfg);
    ASSERT_NE(sub, nullptr);
    snn_network_attach_substrate(network, sub);

    HardReset(pop);

    /* Moderately strong drive so normal firing emerges. */
    uint64_t total = 0;
    const int n_steps = 250;
    for (int step = 0; step < n_steps; step++) {
        InjectCurrent(pop, 40.0f);
        int rc = snn_network_step(network, 0.1f);
        ASSERT_GE(rc, 0) << "Step " << step << " returned error.";
        total += CountSpikes(pop);

        /* Sanity: no NaN/Inf in membrane potential. Sample 5 neurons. */
        float* v = (float*)nimcp_tensor_data(pop->membrane_v);
        if (v) {
            for (uint32_t k = 0; k < pop->n_neurons && k < 5; k++) {
                ASSERT_FALSE(std::isnan(v[k])) << "NaN v[" << k << "] step=" << step;
                ASSERT_FALSE(std::isinf(v[k])) << "Inf v[" << k << "] step=" << step;
            }
        }
    }
    EXPECT_GT(total, 0u) << "Full-stack run produced zero spikes.";
}

/*============================================================================
 * Test 3: Update period — effects cache refreshes only every N steps.
 * Set period=50, alter substrate.atp between steps 2 and 10, and verify the
 * cache still reflects the OLD atp (cache was filled at step 1 and won't
 * refresh again until step 51). Biophysics off to isolate.
 *==========================================================================*/
TEST_F(SNNSubstrateIntegrationTest, UpdatePeriodRateLimitsCacheRefresh) {
    snn_tune_set_ahp_enabled(0.0f);
    snn_tune_set_pump_enabled(0.0f);
    snn_tune_set_basket_enabled(0.0f);
    snn_tune_set_noise_rate_hz(0.0f);

    snn_tune_set_substrate_enabled(1.0f);
    snn_tune_set_substrate_update_period(50.0f);

    snn_population_t* pop = BuildLightweight(10);
    ASSERT_NE(pop, nullptr);

    substrate_config_t scfg;
    substrate_default_config(&scfg);
    sub = substrate_create(&scfg);
    ASSERT_NE(sub, nullptr);
    substrate_set_atp(sub, 1.0f);
    substrate_set_temperature(sub, 37.0f);
    snn_network_attach_substrate(network, sub);

    /* Step 1: cache populates with atp=1.0. */
    InjectCurrent(pop, 30.0f);
    ASSERT_GE(snn_network_step(network, 0.1f), 0);
    ASSERT_NEAR(network->cached_axon_effects.atp_velocity_factor, 1.0f, 1e-4f);

    /* Change substrate atp on the fly. */
    substrate_set_atp(sub, 0.2f);

    /* Step more — but period is 50, so we're still within the window
     * where the cache stays at its first-step value. */
    for (int i = 0; i < 20; i++) {
        InjectCurrent(pop, 30.0f);
        ASSERT_GE(snn_network_step(network, 0.1f), 0);
    }
    EXPECT_NEAR(network->cached_axon_effects.atp_velocity_factor, 1.0f, 1e-4f)
        << "Cache should not have refreshed yet (period=50).";

    /* Run past the period boundary. Total steps so far = 21. Next 30 more
     * steps puts us at 51 total — cache must refresh on step 51. */
    for (int i = 0; i < 30; i++) {
        InjectCurrent(pop, 30.0f);
        ASSERT_GE(snn_network_step(network, 0.1f), 0);
    }
    /* Now the cache should reflect atp=0.2. */
    EXPECT_NEAR(network->cached_axon_effects.atp_velocity_factor, 0.2f, 1e-3f)
        << "Cache should have refreshed with new atp after period boundary.";
}
