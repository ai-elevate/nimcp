//=============================================================================
// test_signal_filter_integration.cpp - Signal Filter Integration Tests
//=============================================================================
/**
 * @file test_signal_filter_integration.cpp
 * @brief Integration tests for signal filter in oscillation detector
 *
 * WHAT: Tests signal filter + Hilbert + oscillation detector integration
 * WHY:  Verify filter correctly isolates frequency bands for accurate detection
 * HOW:  Generate multi-band signals, filter, detect, validate band separation
 *
 * TEST COVERAGE:
 * - All 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
 * - Filter + Hilbert pipeline accuracy
 * - Multi-band signal decomposition
 * - Filter transient handling
 * - Performance: DFT vs Hilbert+filter methods
 * - Edge cases: filter fallback, boundary frequencies
 */

#include <gtest/gtest.h>
extern "C" {
    #include "utils/signal/nimcp_signal_filter.h"
    #include "utils/signal/nimcp_hilbert.h"
    #include "middleware/patterns/nimcp_oscillation_detector.h"
    #include "utils/memory/nimcp_memory.h"
}
#include <cmath>
#include <vector>
#include <algorithm>

#define M_PI 3.14159265358979323846
#define TOLERANCE 0.05f  // 5% tolerance for band power
#define PHASE_TOLERANCE 0.1f

//=============================================================================
// Test Fixture
//=============================================================================

class SignalFilterIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize detector with Hilbert method
        config = oscillation_detector_default_config();
        config.use_phasor_detection = true;  // Use Hilbert+filter method
        config.enable_pac = true;
        config.enable_plv = true;

        detector = oscillation_detector_create(&config);
        ASSERT_NE(detector, nullptr);
    }

    void TearDown() override {
        if (detector) {
            oscillation_detector_destroy(detector);
        }
    }

    // Helper: Generate pure tone at specified frequency
    std::vector<float> generate_pure_tone(
        uint32_t num_samples,
        float frequency,
        float amplitude = 1.0f
    ) {
        std::vector<float> signal(num_samples);
        const float sample_rate = config.sample_rate_hz;

        for (uint32_t i = 0; i < num_samples; i++) {
            float t = (float)i / sample_rate;
            signal[i] = amplitude * sinf(2.0f * M_PI * frequency * t);
        }

        return signal;
    }

    // Helper: Generate multi-band signal
    std::vector<float> generate_multiband_signal(
        uint32_t num_samples,
        const std::vector<std::pair<float, float>>& freq_amp_pairs
    ) {
        std::vector<float> signal(num_samples, 0.0f);
        const float sample_rate = config.sample_rate_hz;

        for (const auto& pair : freq_amp_pairs) {
            float freq = pair.first;
            float amp = pair.second;

            for (uint32_t i = 0; i < num_samples; i++) {
                float t = (float)i / sample_rate;
                signal[i] += amp * sinf(2.0f * M_PI * freq * t);
            }
        }

        return signal;
    }

    // Helper: Compute RMS power
    float compute_rms(const std::vector<float>& signal, uint32_t skip_samples = 100) {
        if (signal.size() <= 2 * skip_samples) return 0.0f;

        float sum = 0.0f;
        uint32_t count = 0;

        for (uint32_t i = skip_samples; i < signal.size() - skip_samples; i++) {
            sum += signal[i] * signal[i];
            count++;
        }

        return sqrtf(sum / (float)count);
    }

    oscillation_detector_config_t config;
    oscillation_detector_t* detector;
};

//=============================================================================
// Individual Band Detection Tests
//=============================================================================

TEST_F(SignalFilterIntegrationTest, DeltaBandDetection) {
    // Generate 2Hz pure tone (Delta: 0-4Hz)
    auto signal = generate_pure_tone(1024, 2.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Delta band should be dominant
    EXPECT_EQ(result.dominant_band, OSC_BAND_DELTA);

    // Delta should have highest power
    EXPECT_GT(result.bands[OSC_BAND_DELTA].power,
              result.bands[OSC_BAND_THETA].power);
    EXPECT_GT(result.bands[OSC_BAND_DELTA].power,
              result.bands[OSC_BAND_ALPHA].power);
    EXPECT_GT(result.bands[OSC_BAND_DELTA].power,
              result.bands[OSC_BAND_BETA].power);
    EXPECT_GT(result.bands[OSC_BAND_DELTA].power,
              result.bands[OSC_BAND_GAMMA].power);
}

TEST_F(SignalFilterIntegrationTest, ThetaBandDetection) {
    // Generate 6Hz pure tone (Theta: 4-8Hz)
    auto signal = generate_pure_tone(1024, 6.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Theta band should be dominant
    EXPECT_EQ(result.dominant_band, OSC_BAND_THETA);

    // Theta should have highest power
    EXPECT_GT(result.bands[OSC_BAND_THETA].power,
              result.bands[OSC_BAND_DELTA].power);
    EXPECT_GT(result.bands[OSC_BAND_THETA].power,
              result.bands[OSC_BAND_ALPHA].power);
}

TEST_F(SignalFilterIntegrationTest, AlphaBandDetection) {
    // Generate 10Hz pure tone (Alpha: 8-13Hz)
    auto signal = generate_pure_tone(1024, 10.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Alpha band should be dominant
    EXPECT_EQ(result.dominant_band, OSC_BAND_ALPHA);

    // Alpha should have highest power
    EXPECT_GT(result.bands[OSC_BAND_ALPHA].power,
              result.bands[OSC_BAND_THETA].power);
    EXPECT_GT(result.bands[OSC_BAND_ALPHA].power,
              result.bands[OSC_BAND_BETA].power);
}

TEST_F(SignalFilterIntegrationTest, BetaBandDetection) {
    // Generate 20Hz pure tone (Beta: 13-30Hz)
    auto signal = generate_pure_tone(1024, 20.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Beta band should be dominant
    EXPECT_EQ(result.dominant_band, OSC_BAND_BETA);

    // Beta should have highest power
    EXPECT_GT(result.bands[OSC_BAND_BETA].power,
              result.bands[OSC_BAND_ALPHA].power);
    EXPECT_GT(result.bands[OSC_BAND_BETA].power,
              result.bands[OSC_BAND_GAMMA].power);
}

TEST_F(SignalFilterIntegrationTest, GammaBandDetection) {
    // Generate 40Hz pure tone (Gamma: 30-100Hz)
    auto signal = generate_pure_tone(1024, 40.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Gamma band should be dominant
    EXPECT_EQ(result.dominant_band, OSC_BAND_GAMMA);

    // Gamma should have highest power
    EXPECT_GT(result.bands[OSC_BAND_GAMMA].power,
              result.bands[OSC_BAND_BETA].power);
    EXPECT_GT(result.bands[OSC_BAND_GAMMA].power,
              result.bands[OSC_BAND_ALPHA].power);
}

//=============================================================================
// Multi-band Signal Decomposition Tests
//=============================================================================

TEST_F(SignalFilterIntegrationTest, MultibandDecomposition_ThetaGamma) {
    // Generate signal with theta (6Hz) and gamma (40Hz) components
    auto signal = generate_multiband_signal(1024, {
        {6.0f, 1.0f},   // Theta
        {40.0f, 0.5f}   // Gamma (half amplitude)
    });

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Both theta and gamma should have significant power
    EXPECT_GT(result.bands[OSC_BAND_THETA].power, 0.3f);
    EXPECT_GT(result.bands[OSC_BAND_GAMMA].power, 0.1f);

    // Theta should have more power (higher amplitude)
    EXPECT_GT(result.bands[OSC_BAND_THETA].power,
              result.bands[OSC_BAND_GAMMA].power);

    // Other bands should have minimal power
    EXPECT_LT(result.bands[OSC_BAND_DELTA].power, 0.1f);
    EXPECT_LT(result.bands[OSC_BAND_BETA].power, 0.2f);
}

TEST_F(SignalFilterIntegrationTest, MultibandDecomposition_AllBands) {
    // Generate signal with all 5 bands
    auto signal = generate_multiband_signal(2048, {
        {2.0f, 0.5f},   // Delta
        {6.0f, 1.0f},   // Theta (strongest)
        {10.0f, 0.7f},  // Alpha
        {20.0f, 0.6f},  // Beta
        {40.0f, 0.4f}   // Gamma
    });

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // All bands should have non-zero power
    EXPECT_GT(result.bands[OSC_BAND_DELTA].power, 0.05f);
    EXPECT_GT(result.bands[OSC_BAND_THETA].power, 0.1f);
    EXPECT_GT(result.bands[OSC_BAND_ALPHA].power, 0.1f);
    EXPECT_GT(result.bands[OSC_BAND_BETA].power, 0.05f);
    EXPECT_GT(result.bands[OSC_BAND_GAMMA].power, 0.05f);

    // Theta should be dominant (highest amplitude)
    EXPECT_EQ(result.dominant_band, OSC_BAND_THETA);
}

//=============================================================================
// Filter Accuracy Tests
//=============================================================================

TEST_F(SignalFilterIntegrationTest, FilterAccuracy_BandRejection) {
    // Generate 40Hz signal, verify other bands reject it
    auto signal = generate_pure_tone(1024, 40.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Gamma should capture most power
    float gamma_power = result.bands[OSC_BAND_GAMMA].power;
    EXPECT_GT(gamma_power, 0.5f);

    // Other bands should have <10% of gamma power (filter rejection)
    EXPECT_LT(result.bands[OSC_BAND_DELTA].power, gamma_power * 0.1f);
    EXPECT_LT(result.bands[OSC_BAND_THETA].power, gamma_power * 0.1f);
    EXPECT_LT(result.bands[OSC_BAND_ALPHA].power, gamma_power * 0.1f);
    EXPECT_LT(result.bands[OSC_BAND_BETA].power, gamma_power * 0.2f);  // Beta closer to Gamma
}

TEST_F(SignalFilterIntegrationTest, FilterAccuracy_BoundaryFrequencies) {
    // Test frequencies at band boundaries

    // 4Hz - boundary between Delta and Theta
    auto signal_4hz = generate_pure_tone(1024, 4.0f);
    oscillation_detector_t* det_4hz = oscillation_detector_create(&config);

    for (size_t i = 0; i < signal_4hz.size(); i++) {
        oscillation_detector_add_sample(det_4hz, signal_4hz[i], (double)i);
    }

    oscillation_result_t result_4hz;
    ASSERT_TRUE(oscillation_detector_detect(det_4hz, &result_4hz));

    // Both Delta and Theta should capture energy at boundary
    EXPECT_GT(result_4hz.bands[OSC_BAND_DELTA].power, 0.1f);
    EXPECT_GT(result_4hz.bands[OSC_BAND_THETA].power, 0.1f);

    oscillation_detector_destroy(det_4hz);

    // 8Hz - boundary between Theta and Alpha
    auto signal_8hz = generate_pure_tone(1024, 8.0f);
    oscillation_detector_t* det_8hz = oscillation_detector_create(&config);

    for (size_t i = 0; i < signal_8hz.size(); i++) {
        oscillation_detector_add_sample(det_8hz, signal_8hz[i], (double)i);
    }

    oscillation_result_t result_8hz;
    ASSERT_TRUE(oscillation_detector_detect(det_8hz, &result_8hz));

    // Both Theta and Alpha should capture energy at boundary
    EXPECT_GT(result_8hz.bands[OSC_BAND_THETA].power, 0.1f);
    EXPECT_GT(result_8hz.bands[OSC_BAND_ALPHA].power, 0.1f);

    oscillation_detector_destroy(det_8hz);
}

//=============================================================================
// Transient Handling Tests
//=============================================================================

TEST_F(SignalFilterIntegrationTest, FilterTransients_ShortSignal) {
    // Test with short signal (256 samples) to check transient impact
    auto signal = generate_pure_tone(256, 40.0f);

    oscillation_detector_t* short_detector = oscillation_detector_create(&config);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(short_detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(short_detector, &result));

    // Gamma should still be detected despite transients
    EXPECT_EQ(result.dominant_band, OSC_BAND_GAMMA);
    EXPECT_GT(result.bands[OSC_BAND_GAMMA].power, 0.3f);

    oscillation_detector_destroy(short_detector);
}

TEST_F(SignalFilterIntegrationTest, FilterTransients_LongSignal) {
    // Test with long signal (4096 samples) - transients should be negligible
    auto signal = generate_pure_tone(4096, 6.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Theta should be clearly dominant with minimal transient impact
    EXPECT_EQ(result.dominant_band, OSC_BAND_THETA);
    EXPECT_GT(result.bands[OSC_BAND_THETA].power, 0.7f);
}

//=============================================================================
// Performance Comparison Tests
//=============================================================================

TEST_F(SignalFilterIntegrationTest, Performance_HilbertVsDFT_Gamma) {
    auto signal = generate_pure_tone(1024, 40.0f);

    // Test with Hilbert+filter method
    oscillation_detector_config_t hilbert_config = oscillation_detector_default_config();
    hilbert_config.use_phasor_detection = true;
    oscillation_detector_t* hilbert_detector = oscillation_detector_create(&hilbert_config);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(hilbert_detector, signal[i], (double)i);
    }

    oscillation_result_t hilbert_result;
    ASSERT_TRUE(oscillation_detector_detect(hilbert_detector, &hilbert_result));

    // Test with DFT method
    oscillation_detector_config_t dft_config = oscillation_detector_default_config();
    dft_config.use_phasor_detection = false;
    oscillation_detector_t* dft_detector = oscillation_detector_create(&dft_config);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(dft_detector, signal[i], (double)i);
    }

    oscillation_result_t dft_result;
    ASSERT_TRUE(oscillation_detector_detect(dft_detector, &dft_result));

    // Both methods should detect Gamma as dominant
    EXPECT_EQ(hilbert_result.dominant_band, OSC_BAND_GAMMA);
    EXPECT_EQ(dft_result.dominant_band, OSC_BAND_GAMMA);

    // Power values should be similar (within 20% tolerance for Gamma)
    float power_ratio = hilbert_result.bands[OSC_BAND_GAMMA].power /
                        dft_result.bands[OSC_BAND_GAMMA].power;
    EXPECT_GT(power_ratio, 0.8f);
    EXPECT_LT(power_ratio, 1.2f);

    oscillation_detector_destroy(hilbert_detector);
    oscillation_detector_destroy(dft_detector);
}

//=============================================================================
// PAC Detection with Filtering
//=============================================================================

TEST_F(SignalFilterIntegrationTest, PAC_ThetaGammaCoupling) {
    // Generate theta-gamma coupled signal
    const uint32_t n = 2048;
    const float sample_rate = config.sample_rate_hz;
    const float theta_freq = 6.0f;
    const float gamma_freq = 40.0f;
    const float coupling_strength = 0.8f;

    std::vector<float> signal(n);

    for (uint32_t i = 0; i < n; i++) {
        float t = (float)i / sample_rate;
        float theta_phase = 2.0f * M_PI * theta_freq * t;

        // Gamma amplitude modulated by theta phase
        float modulation = coupling_strength * (1.0f + cosf(theta_phase)) +
                          (1.0f - coupling_strength);

        signal[i] = 0.5f * sinf(theta_phase) +  // Theta carrier
                    modulation * sinf(2.0f * M_PI * gamma_freq * t);  // Modulated gamma
    }

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Detect PAC
    cross_freq_coupling_t couplings[10];
    uint32_t num_found = 0;
    ASSERT_TRUE(oscillation_detector_detect_pac(detector, couplings, 10, &num_found));

    // PAC should be detected
    EXPECT_GT(num_found, 0u);

    // Find theta-gamma coupling
    bool found_theta_gamma = false;
    for (uint32_t i = 0; i < num_found; i++) {
        if (couplings[i].phase_band == OSC_BAND_THETA &&
            couplings[i].amp_band == OSC_BAND_GAMMA) {
            found_theta_gamma = true;
            // Coupling strength should be significant (> 0.1 for strong coupling)
            EXPECT_GT(couplings[i].coupling_strength, 0.1f);
            EXPECT_LT(couplings[i].coupling_strength, 1.0f);
            break;
        }
    }
    EXPECT_TRUE(found_theta_gamma);
}

//=============================================================================
// Edge Cases and Error Handling
//=============================================================================

TEST_F(SignalFilterIntegrationTest, EdgeCase_ZeroSignal) {
    // Test with zero signal
    std::vector<float> signal(1024, 0.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // All bands should have near-zero power
    for (int band = 0; band < OSC_NUM_BANDS; band++) {
        EXPECT_LT(result.bands[band].power, 0.01f);
    }
}

TEST_F(SignalFilterIntegrationTest, EdgeCase_DCOffset) {
    // Test signal with DC offset (should be filtered out by bandpass)
    auto signal = generate_pure_tone(1024, 40.0f);

    // Add DC offset
    for (auto& sample : signal) {
        sample += 5.0f;
    }

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // Gamma should still be detected (DC offset filtered)
    EXPECT_EQ(result.dominant_band, OSC_BAND_GAMMA);
    EXPECT_GT(result.bands[OSC_BAND_GAMMA].power, 0.5f);
}

TEST_F(SignalFilterIntegrationTest, EdgeCase_HighFrequency) {
    // Test with frequency above Gamma range (150Hz)
    auto signal = generate_pure_tone(1024, 150.0f);

    for (size_t i = 0; i < signal.size(); i++) {
        oscillation_detector_add_sample(detector, signal[i], (double)i);
    }

    oscillation_result_t result;
    ASSERT_TRUE(oscillation_detector_detect(detector, &result));

    // 150Hz is outside all band ranges - power should be low across all bands
    for (int band = 0; band < OSC_NUM_BANDS; band++) {
        EXPECT_LT(result.bands[band].power, 0.3f);
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
