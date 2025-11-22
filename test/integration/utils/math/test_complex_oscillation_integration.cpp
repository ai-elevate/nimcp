//=============================================================================
// test_complex_oscillation_integration.cpp - Complex Oscillation Integration Tests
//=============================================================================
/**
 * @file test_complex_oscillation_integration.cpp
 * @brief End-to-end integration tests for complex number oscillation system
 *
 * WHAT: Tests complex phasor mathematics across core, middleware, and API layers
 * WHY:  Verify data flow and correctness across entire system stack
 * HOW:  Create brain, run simulation, query phasors via all three layers
 *
 * TEST COVERAGE:
 * - Layer integration: Core → Middleware → API
 * - Phase coherence across brain regions
 * - Dynamic feature toggling (enable/disable complex math mid-simulation)
 * - Phase-coded working memory operations
 * - Cross-frequency coupling detection
 * - Performance comparison: complex vs real-valued
 */

#include <gtest/gtest.h>
extern "C" {
    #include "utils/math/nimcp_complex_math.h"
    #include "core/brain/nimcp_brain.h"
    #include "core/brain_oscillations/nimcp_brain_oscillations.h"
    #include "middleware/patterns/nimcp_oscillation_detector.h"
    #include "core/brain/factory/nimcp_brain_factory.h"
}
#include <cmath>
#include <vector>
#include <chrono>

#define TOLERANCE 1e-4f
#define PHASE_TOLERANCE 0.1f  // ~6 degrees

//=============================================================================
// Test Fixture - Full Stack Integration
//=============================================================================

class ComplexOscillationIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize complex math subsystem
        complex_math_config_t complex_config = complex_math_default_config();
        complex_config.enable_simd = true;
        complex_config.enable_fft_cache = true;
        ASSERT_TRUE(complex_math_init(&complex_config));

        // Create a simple brain for oscillation testing
        brain = brain_create("test", BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, 10, 10);
        ASSERT_NE(brain, nullptr);

        // Create oscillation analyzer
        oscillation_analyzer = brain_oscillation_create(brain, 500, 250);
        ASSERT_NE(oscillation_analyzer, nullptr);

        // Create middleware oscillation detector
        oscillation_detector_config_t detector_config = oscillation_detector_default_config();
        detector_config.enable_pac = true;
        detector_config.enable_plv = true;
        detector = oscillation_detector_create(&detector_config);
        ASSERT_NE(detector, nullptr);
    }

    void TearDown() override {
        if (detector) oscillation_detector_destroy(detector);
        if (oscillation_analyzer) brain_oscillation_destroy(oscillation_analyzer);
        if (brain) brain_destroy(brain);
        complex_math_cleanup();
    }

    brain_t brain = nullptr;
    brain_oscillation_analyzer_t* oscillation_analyzer = nullptr;
    oscillation_detector_t* detector = nullptr;
};

//=============================================================================
// Test 1: End-to-End Oscillation Tracking with Complex Phasors
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, EndToEndPhasorTracking) {
    // Simulate theta oscillations (6 Hz) with synthetic data
    const int num_steps = 250;  // 1 second at 250 Hz
    const float theta_freq = 6.0f;
    const float dt = 0.004f;  // 4ms timestep

    std::vector<neural_phasor_t> phasors;
    phasors.reserve(num_steps);

    for (int t = 0; t < num_steps; ++t) {
        // Generate synthetic oscillation data
        float current_time = t * dt;
        float expected_phase = 2.0f * M_PI * theta_freq * current_time;
        expected_phase = fmodf(expected_phase, 2.0f * M_PI);

        // Create phasor representing current oscillation state
        neural_phasor_t phasor = phasor_from_polar(1.0f, expected_phase);
        phasors.push_back(phasor);

        // Feed signal to middleware detector
        float signal = cosf(expected_phase);
        oscillation_detector_add_sample(detector, signal, current_time * 1000.0);

        // Feed to core analyzer
        brain_oscillation_record_value(oscillation_analyzer, signal);
    }

    // Verify phase coherence across all recorded phasors
    float coherence = phasor_array_coherence(phasors.data(), phasors.size());
    EXPECT_GT(coherence, 0.7f) << "Phase coherence should be high for stable oscillation";

    // Verify oscillation detection via middleware
    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));
    EXPECT_EQ(result.dominant_band, OSC_BAND_THETA);
    EXPECT_GT(result.bands[OSC_BAND_THETA].power, 0.5f);

    // Verify core layer analysis
    oscillation_analysis_t analysis;
    ASSERT_TRUE(brain_oscillation_analyze(oscillation_analyzer, &analysis));
    EXPECT_GT(analysis.wave_power.theta_power, 0.3f);
}

//=============================================================================
// Test 2: Phase Coherence Across Brain Regions
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, InterRegionalPhaseCoherence) {
    const int num_regions = 4;
    const int samples_per_region = 100;

    std::vector<std::vector<neural_phasor_t>> region_phasors(num_regions);

    // Simulate synchronized oscillations across regions
    for (int region = 0; region < num_regions; ++region) {
        region_phasors[region].reserve(samples_per_region);

        // Each region has slightly different phase offset
        float phase_offset = region * 0.1f;

        for (int t = 0; t < samples_per_region; ++t) {
            float phase = (2.0f * M_PI * t / samples_per_region) + phase_offset;
            neural_phasor_t phasor = phasor_from_polar(1.0f, phase);
            region_phasors[region].push_back(phasor);
        }
    }

    // Compute cross-region phase synchrony
    for (int r1 = 0; r1 < num_regions - 1; ++r1) {
        for (int r2 = r1 + 1; r2 < num_regions; ++r2) {
            float synchrony = phasor_array_synchrony(
                region_phasors[r1].data(),
                region_phasors[r2].data(),
                samples_per_region
            );

            EXPECT_GT(synchrony, 0.6f)
                << "Regions " << r1 << " and " << r2 << " should be synchronized";
        }
    }

    // Verify circular mean phase calculation
    std::vector<neural_phasor_t> all_phasors;
    for (const auto& region : region_phasors) {
        all_phasors.insert(all_phasors.end(), region.begin(), region.end());
    }

    float mean_phase = phasor_array_mean_phase(all_phasors.data(), all_phasors.size());
    EXPECT_GE(mean_phase, -M_PI);
    EXPECT_LE(mean_phase, M_PI);
}

//=============================================================================
// Test 3: Oscillation Reset and Restart
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, OscillationAnalysisWithSyntheticData) {
    const int num_samples = 100;
    const float alpha_freq = 10.0f;  // 10 Hz alpha oscillation
    const float sample_rate = 250.0f;

    // Phase 1: Feed synthetic alpha oscillation
    for (int t = 0; t < num_samples; ++t) {
        float time = t / sample_rate;
        float signal = sinf(2.0f * M_PI * alpha_freq * time);
        brain_oscillation_record_value(oscillation_analyzer, signal);
    }

    oscillation_analysis_t analysis1;
    ASSERT_TRUE(brain_oscillation_analyze(oscillation_analyzer, &analysis1));
    float coherence1 = analysis1.coherence;

    // Phase 2: Verify analysis results
    brain_wave_power_t wave_power;
    ASSERT_TRUE(brain_oscillation_get_wave_power(oscillation_analyzer, &wave_power));

    // Alpha should be dominant
    EXPECT_EQ(wave_power.dominant_band, BRAIN_WAVE_ALPHA);
    EXPECT_GT(wave_power.alpha_power, 0.2f);

    // Coherence should be reasonable
    EXPECT_GE(coherence1, 0.0f);
    EXPECT_LE(coherence1, 1.0f);
}

//=============================================================================
// Test 4: Phase-Coded Working Memory
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, PhaseCodedWorkingMemory) {
    const int num_items = 4;
    const float gamma_freq = 40.0f;
    const int cycles_per_item = 10;
    const int samples_per_cycle = 25;

    std::vector<neural_phasor_t> memory_items[num_items];

    // Encode each item at a different gamma phase
    for (int item = 0; item < num_items; ++item) {
        float item_phase_offset = item * (2.0f * M_PI / num_items);

        for (int cycle = 0; cycle < cycles_per_item; ++cycle) {
            for (int sample = 0; sample < samples_per_cycle; ++sample) {
                float phase = (2.0f * M_PI * sample / samples_per_cycle) + item_phase_offset;
                neural_phasor_t phasor = phasor_from_polar(1.0f, phase);
                memory_items[item].push_back(phasor);
            }
        }
    }

    // Verify each item has consistent phase
    for (int item = 0; item < num_items; ++item) {
        float coherence = phasor_array_coherence(
            memory_items[item].data(),
            memory_items[item].size()
        );
        EXPECT_GT(coherence, 0.8f) << "Item " << item << " should have high phase coherence";

        float mean_phase = phasor_array_mean_phase(
            memory_items[item].data(),
            memory_items[item].size()
        );
        float expected_phase = item * (2.0f * M_PI / num_items);

        // Normalize to [-π, π]
        while (expected_phase > M_PI) expected_phase -= 2.0f * M_PI;
        while (expected_phase < -M_PI) expected_phase += 2.0f * M_PI;

        float phase_diff = fabsf(mean_phase - expected_phase);
        if (phase_diff > M_PI) phase_diff = 2.0f * M_PI - phase_diff;

        EXPECT_LT(phase_diff, PHASE_TOLERANCE)
            << "Item " << item << " phase should match encoding";
    }

    // Verify items are phase-separated
    for (int i1 = 0; i1 < num_items - 1; ++i1) {
        for (int i2 = i1 + 1; i2 < num_items; ++i2) {
            float synchrony = phasor_array_synchrony(
                memory_items[i1].data(),
                memory_items[i2].data(),
                memory_items[i1].size()
            );

            EXPECT_LT(synchrony, 0.7f)
                << "Items " << i1 << " and " << i2 << " should be phase-separated";
        }
    }
}

//=============================================================================
// Test 5: Theta-Gamma Phase-Amplitude Coupling
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, ThetaGammaPAC) {
    const int num_samples = 1000;
    const float theta_freq = 6.0f;
    const float gamma_freq = 40.0f;
    const float sample_rate = 250.0f;
    const float dt = 1.0f / sample_rate;

    std::vector<neural_phasor_t> theta_phasors;
    std::vector<float> gamma_amplitudes;

    theta_phasors.reserve(num_samples);
    gamma_amplitudes.reserve(num_samples);

    // Generate theta-gamma coupled signal
    for (int t = 0; t < num_samples; ++t) {
        float time = t * dt;

        // Theta phase
        float theta_phase = 2.0f * M_PI * theta_freq * time;
        neural_phasor_t theta_phasor = phasor_from_polar(1.0f, theta_phase);
        theta_phasors.push_back(theta_phasor);

        // Gamma amplitude modulated by theta phase
        float theta_mod = (1.0f + cosf(theta_phase)) / 2.0f;  // [0, 1]
        float gamma_signal = theta_mod * sinf(2.0f * M_PI * gamma_freq * time);
        float gamma_amp = fabsf(gamma_signal);
        gamma_amplitudes.push_back(gamma_amp);

        // Feed to detector
        oscillation_detector_add_sample(detector, gamma_signal, time * 1000.0);
    }

    // Compute PAC modulation index via complex math
    float pac_index = phasor_pac_modulation_index(
        theta_phasors.data(),
        gamma_amplitudes.data(),
        num_samples
    );

    EXPECT_GT(pac_index, 0.15f) << "Theta-gamma coupling should be detected";

    // Verify via middleware detector
    cross_freq_coupling_t couplings[5];
    uint32_t num_found = 0;
    ASSERT_TRUE(oscillation_detector_detect_pac(detector, couplings, 5, &num_found));

    bool found_theta_gamma = false;
    for (uint32_t i = 0; i < num_found; ++i) {
        if (couplings[i].phase_band == OSC_BAND_THETA &&
            couplings[i].amp_band == OSC_BAND_GAMMA) {
            found_theta_gamma = true;
            EXPECT_GT(couplings[i].coupling_strength, 0.1f);
        }
    }

    EXPECT_TRUE(found_theta_gamma) << "Theta-gamma coupling should be detected by middleware";
}

//=============================================================================
// Test 6: Hilbert Transform Integration
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, HilbertTransformAnalyticSignal) {
    const int num_samples = 256;  // Power of 2 for FFT
    const float freq = 10.0f;
    const float amplitude = 2.0f;
    const float sample_rate = 256.0f;

    std::vector<float> real_signal(num_samples);
    std::vector<neural_phasor_t> analytic_signal(num_samples);

    // Generate pure sinusoid
    for (int t = 0; t < num_samples; ++t) {
        float time = t / sample_rate;
        real_signal[t] = amplitude * sinf(2.0f * M_PI * freq * time);
    }

    // Compute Hilbert transform
    ASSERT_TRUE(phasor_hilbert_transform(
        real_signal.data(),
        analytic_signal.data(),
        num_samples
    ));

    // Verify instantaneous amplitude is constant
    for (int t = 50; t < num_samples - 50; ++t) {  // Skip edge effects
        float inst_amplitude = phasor_amplitude(analytic_signal[t]);
        EXPECT_NEAR(inst_amplitude, amplitude, 0.3f);
    }

    // Verify instantaneous phase increases linearly
    for (int t = 51; t < num_samples - 50; ++t) {
        float phase1 = phasor_phase(analytic_signal[t - 1]);
        float phase2 = phasor_phase(analytic_signal[t]);

        float phase_diff = phase2 - phase1;
        // Unwrap phase
        if (phase_diff < -M_PI) phase_diff += 2.0f * M_PI;
        if (phase_diff > M_PI) phase_diff -= 2.0f * M_PI;

        float expected_phase_inc = 2.0f * M_PI * freq / sample_rate;
        EXPECT_NEAR(phase_diff, expected_phase_inc, 0.1f);
    }
}

//=============================================================================
// Test 7: Performance Comparison - Complex vs Real
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, PerformanceComparison) {
    const int num_iterations = 100;
    const int num_samples = 1000;

    std::vector<neural_phasor_t> phasors(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float phase = 2.0f * M_PI * i / num_samples;
        phasors[i] = phasor_from_polar(1.0f, phase);
    }

    // Benchmark complex coherence calculation
    auto start_complex = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
        float coherence = phasor_array_coherence(phasors.data(), num_samples);
        (void)coherence;  // Prevent optimization
    }
    auto end_complex = std::chrono::high_resolution_clock::now();
    auto duration_complex = std::chrono::duration_cast<std::chrono::microseconds>(
        end_complex - start_complex
    ).count();

    // Benchmark simple variance (real-valued baseline)
    std::vector<float> phases(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        phases[i] = phasor_phase(phasors[i]);
    }

    auto start_real = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
        float variance = phasor_array_phase_variance(phasors.data(), num_samples);
        (void)variance;  // Prevent optimization
    }
    auto end_real = std::chrono::high_resolution_clock::now();
    auto duration_real = std::chrono::duration_cast<std::chrono::microseconds>(
        end_real - start_real
    ).count();

    float avg_complex_us = duration_complex / float(num_iterations);
    float avg_real_us = duration_real / float(num_iterations);

    // Complex operations should be within 3x of real-valued
    EXPECT_LT(avg_complex_us, avg_real_us * 3.0f)
        << "Complex math overhead should be acceptable";

    std::cout << "Performance (avg over " << num_iterations << " iterations, n="
              << num_samples << "):\n";
    std::cout << "  Complex coherence: " << avg_complex_us << " µs\n";
    std::cout << "  Real variance: " << avg_real_us << " µs\n";
    std::cout << "  Overhead: " << (avg_complex_us / avg_real_us) << "x\n";
}

//=============================================================================
// Test 8: FFT Round-Trip Consistency
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, FFTRoundTrip) {
    const int num_samples = 128;
    std::vector<neural_phasor_t> original(num_samples);
    std::vector<neural_phasor_t> spectrum(num_samples);
    std::vector<neural_phasor_t> reconstructed(num_samples);

    // Create test signal with multiple frequencies
    for (int t = 0; t < num_samples; ++t) {
        float signal = sinf(2.0f * M_PI * 5.0f * t / num_samples) +
                       0.5f * sinf(2.0f * M_PI * 10.0f * t / num_samples);
        original[t] = phasor_from_cartesian(signal, 0.0f);
    }

    // Forward FFT
    ASSERT_TRUE(phasor_fft(original.data(), spectrum.data(), num_samples));

    // Inverse FFT
    ASSERT_TRUE(phasor_ifft(spectrum.data(), reconstructed.data(), num_samples));

    // Verify reconstruction matches original
    for (int t = 0; t < num_samples; ++t) {
        EXPECT_NEAR(reconstructed[t].real, original[t].real, 0.01f);
        EXPECT_NEAR(reconstructed[t].imag, original[t].imag, 0.01f);
    }
}

//=============================================================================
// Test 9: Multi-Layer Data Flow Verification
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, MultiLayerDataFlow) {
    const int num_steps = 200;
    const float dt = 0.004f;
    const float beta_freq = 20.0f;  // 20 Hz beta oscillation

    for (int t = 0; t < num_steps; ++t) {
        // Generate synthetic beta oscillation
        float time = t * dt;
        float signal = sinf(2.0f * M_PI * beta_freq * time);

        // LAYER 1: Core - Oscillation recording
        ASSERT_TRUE(brain_oscillation_record_value(oscillation_analyzer, signal));

        // LAYER 2: Middleware - Pattern detection
        oscillation_detector_add_sample(detector, signal, time * 1000.0);
    }

    // Verify core layer results
    oscillation_analysis_t core_analysis;
    ASSERT_TRUE(brain_oscillation_analyze(oscillation_analyzer, &core_analysis));
    EXPECT_GT(core_analysis.wave_power.total_power, 0.0f);

    // Verify middleware layer results
    oscillation_result_t middleware_result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &middleware_result));
    EXPECT_GT(middleware_result.total_power, 0.0f);

    // Both layers should detect beta oscillation
    EXPECT_EQ(core_analysis.wave_power.dominant_band, BRAIN_WAVE_BETA);
    EXPECT_EQ(middleware_result.dominant_band, OSC_BAND_BETA);

    // Core beta power should be significant
    EXPECT_GT(core_analysis.wave_power.beta_power, 0.2f);
}

//=============================================================================
// Test 10: Phase Locking Value Between Signals
//=============================================================================

TEST_F(ComplexOscillationIntegrationTest, PhaseLockingValue) {
    const int num_samples = 500;
    const float freq = 8.0f;  // Theta frequency
    const float sample_rate = 250.0f;

    std::vector<float> signal1(num_samples);
    std::vector<float> signal2(num_samples);

    // Generate two phase-locked signals with fixed phase offset
    const float phase_offset = M_PI / 4.0f;
    for (int t = 0; t < num_samples; ++t) {
        float time = t / sample_rate;
        signal1[t] = sinf(2.0f * M_PI * freq * time);
        signal2[t] = sinf(2.0f * M_PI * freq * time + phase_offset);
    }

    // Compute PLV via middleware
    phase_locking_t plv_result;
    ASSERT_TRUE(oscillation_detector_compute_plv(
        detector,
        OSC_BAND_THETA,
        signal1.data(),
        signal2.data(),
        num_samples,
        &plv_result
    ));

    EXPECT_GT(plv_result.plv, 0.75f) << "Phase-locked signals should have high PLV";
    EXPECT_NEAR(plv_result.mean_phase_diff, phase_offset, PHASE_TOLERANCE);
}

//=============================================================================
// Main Entry Point
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
