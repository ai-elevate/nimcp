/**
 * @file test_audio_cortex.cpp
 * @brief Unit tests for audio cortex
 *
 * WHAT: Test auditory processing and feature extraction
 * WHY:  Ensure audio cortex works correctly
 * HOW:  Unit tests with synthetic audio signals
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "include/perception/nimcp_audio_cortex.h"
#include "utils/memory/nimcp_memory.h"
}

//=============================================================================
// Test Fixtures
//=============================================================================

class AudioCortexTest : public ::testing::Test {
protected:
    audio_cortex_t* cortex;
    audio_cortex_config_t config;

    void SetUp() override {
        config.sample_rate = 16000;
        config.frame_size = 512;
        config.num_freq_bins = 256;
        config.num_mel_filters = 40;
        config.num_mfcc = 13;
        config.num_channels = 1;
        config.feature_dim = 13;
        config.enable_attention = true;
        config.enable_memory = true;

        cortex = audio_cortex_create(&config);
        ASSERT_NE(cortex, nullptr);
    }

    void TearDown() override {
        if (cortex) {
            audio_cortex_destroy(cortex);
        }
    }

    // Helper: Generate sine wave
    std::vector<float> generate_sine_wave(float frequency, uint32_t num_samples, float amplitude = 1.0f) {
        std::vector<float> signal(num_samples);
        float sample_rate = (float)config.sample_rate;

        for (uint32_t i = 0; i < num_samples; i++) {
            float t = (float)i / sample_rate;
            signal[i] = amplitude * sinf(2.0f * M_PI * frequency * t);
        }

        return signal;
    }

    // Helper: Generate white noise
    std::vector<float> generate_white_noise(uint32_t num_samples, float amplitude = 0.1f) {
        std::vector<float> signal(num_samples);

        for (uint32_t i = 0; i < num_samples; i++) {
            signal[i] = amplitude * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        }

        return signal;
    }
};

//=============================================================================
// Basic Functionality Tests
//=============================================================================

TEST_F(AudioCortexTest, CreateDestroy) {
    audio_cortex_config_t test_config = config;
    audio_cortex_t* test_cortex = audio_cortex_create(&test_config);
    ASSERT_NE(test_cortex, nullptr);
    audio_cortex_destroy(test_cortex);
}

TEST_F(AudioCortexTest, InvalidConfig) {
    audio_cortex_config_t invalid_config = config;

    // Invalid sample rate
    invalid_config.sample_rate = 1000;  // Too low
    EXPECT_EQ(audio_cortex_create(&invalid_config), nullptr);

    invalid_config.sample_rate = 100000;  // Too high
    EXPECT_EQ(audio_cortex_create(&invalid_config), nullptr);

    // Invalid channels
    invalid_config = config;
    invalid_config.num_channels = 0;
    EXPECT_EQ(audio_cortex_create(&invalid_config), nullptr);

    invalid_config.num_channels = 10;  // Too many
    EXPECT_EQ(audio_cortex_create(&invalid_config), nullptr);
}

TEST_F(AudioCortexTest, NullInputs) {
    EXPECT_EQ(audio_cortex_create(nullptr), nullptr);

    std::vector<float> signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features(config.feature_dim);

    EXPECT_FALSE(audio_cortex_process(nullptr, signal.data(), config.frame_size, 1, features.data()));
    EXPECT_FALSE(audio_cortex_process(cortex, nullptr, config.frame_size, 1, features.data()));
    EXPECT_FALSE(audio_cortex_process(cortex, signal.data(), config.frame_size, 1, nullptr));
}

//=============================================================================
// Feature Extraction Tests
//=============================================================================

TEST_F(AudioCortexTest, ProcessSineWave) {
    // Generate 440Hz sine wave (A4 note)
    auto signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features(config.feature_dim);

    bool success = audio_cortex_process(cortex, signal.data(), config.frame_size, 1, features.data());
    EXPECT_TRUE(success);

    // Features should be non-zero
    bool has_nonzero = false;
    for (uint32_t i = 0; i < config.feature_dim; i++) {
        if (fabsf(features[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(AudioCortexTest, SpectrumComputation) {
    auto signal = generate_sine_wave(1000.0f, config.frame_size);
    std::vector<float> spectrum(config.num_freq_bins);

    bool success = audio_cortex_compute_spectrum(cortex, signal.data(), config.frame_size, spectrum.data());
    EXPECT_TRUE(success);

    // Spectrum should have energy peak near 1000Hz
    // Find bin corresponding to 1000Hz
    uint32_t expected_bin = (uint32_t)(1000.0f * config.num_freq_bins / (config.sample_rate / 2.0f));

    // Check that peak is near expected frequency
    float max_energy = 0.0f;
    uint32_t max_bin = 0;
    for (uint32_t i = 0; i < config.num_freq_bins; i++) {
        if (spectrum[i] > max_energy) {
            max_energy = spectrum[i];
            max_bin = i;
        }
    }

    EXPECT_NEAR(max_bin, expected_bin, 5);  // Within 5 bins
}

TEST_F(AudioCortexTest, MelFeatures) {
    auto signal = generate_sine_wave(500.0f, config.frame_size);
    std::vector<float> spectrum(config.num_freq_bins);
    std::vector<float> mel_features(config.num_mel_filters);

    audio_cortex_compute_spectrum(cortex, signal.data(), config.frame_size, spectrum.data());
    bool success = audio_cortex_compute_mel_features(
        cortex, spectrum.data(), config.num_freq_bins, mel_features.data()
    );

    EXPECT_TRUE(success);

    // Mel features should be in log scale (negative or small positive values)
    for (uint32_t i = 0; i < config.num_mel_filters; i++) {
        EXPECT_GT(mel_features[i], -50.0f);  // Reasonable log scale range
    }
}

TEST_F(AudioCortexTest, MFCCComputation) {
    auto signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> spectrum(config.num_freq_bins);
    std::vector<float> mel_features(config.num_mel_filters);
    std::vector<float> mfcc(config.num_mfcc);

    audio_cortex_compute_spectrum(cortex, signal.data(), config.frame_size, spectrum.data());
    audio_cortex_compute_mel_features(cortex, spectrum.data(), config.num_freq_bins, mel_features.data());

    bool success = audio_cortex_compute_mfcc(
        cortex, mel_features.data(), config.num_mel_filters, mfcc.data()
    );

    EXPECT_TRUE(success);

    // MFCC coefficients should be finite
    for (uint32_t i = 0; i < config.num_mfcc; i++) {
        EXPECT_TRUE(std::isfinite(mfcc[i]));
    }
}

//=============================================================================
// Attention Tests
//=============================================================================

TEST_F(AudioCortexTest, AttentionMapCreation) {
    audio_attention_map_t* map = audio_attention_map_create(128, 10);
    ASSERT_NE(map, nullptr);
    EXPECT_EQ(map->num_freq, 128);
    EXPECT_EQ(map->num_time, 10);
    EXPECT_NE(map->values, nullptr);
    audio_attention_map_destroy(map);
}

TEST_F(AudioCortexTest, AttentionComputation) {
    auto signal = generate_sine_wave(1000.0f, config.frame_size);
    audio_attention_map_t* map = audio_attention_map_create(config.num_freq_bins, 1);
    ASSERT_NE(map, nullptr);

    bool success = audio_cortex_compute_attention(cortex, signal.data(), config.frame_size, map);
    EXPECT_TRUE(success);

    // Attention should be concentrated near 1000Hz frequency
    uint32_t max_freq, max_time;
    float max_val;
    success = audio_cortex_get_attention_peak(map, &max_freq, &max_time, &max_val);
    EXPECT_TRUE(success);
    EXPECT_GT(max_val, 0.0f);

    audio_attention_map_destroy(map);
}

//=============================================================================
// Memory Tests
//=============================================================================

TEST_F(AudioCortexTest, StoreMemory) {
    auto signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features(config.feature_dim);

    audio_cortex_process(cortex, signal.data(), config.frame_size, 1, features.data());

    bool success = audio_cortex_store_memory(cortex, features.data(), 0.8f);
    EXPECT_TRUE(success);

    audio_cortex_stats_t stats;
    audio_cortex_get_stats(cortex, &stats);
    EXPECT_EQ(stats.memories_stored, 1);
}

TEST_F(AudioCortexTest, RecallMemory) {
    // Store multiple memories
    for (int i = 0; i < 3; i++) {
        float freq = 440.0f + i * 100.0f;
        auto signal = generate_sine_wave(freq, config.frame_size);
        std::vector<float> features(config.feature_dim);

        audio_cortex_process(cortex, signal.data(), config.frame_size, 1, features.data());
        audio_cortex_store_memory(cortex, features.data(), 0.5f + i * 0.1f);
    }

    // Query with similar signal
    auto query_signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> query_features(config.feature_dim);
    audio_cortex_process(cortex, query_signal.data(), config.frame_size, 1, query_features.data());

    auditory_memory_t** recalled = nullptr;
    int num_recalled = 0;
    bool success = audio_cortex_recall_memory(cortex, query_features.data(), 2, &recalled, &num_recalled);

    EXPECT_TRUE(success);
    EXPECT_EQ(num_recalled, 2);
    EXPECT_NE(recalled, nullptr);

    if (recalled) {
        nimcp_free(recalled);
    }
}

//=============================================================================
// Brain Integration Tests
//=============================================================================

TEST_F(AudioCortexTest, NoveltyDetection) {
    // First sound - should be novel
    auto signal1 = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features1(config.feature_dim);
    audio_cortex_process(cortex, signal1.data(), config.frame_size, 1, features1.data());

    float novelty1 = audio_cortex_compute_novelty(cortex, features1.data());
    EXPECT_FLOAT_EQ(novelty1, 1.0f);  // No memories yet

    // Store memory
    audio_cortex_store_memory(cortex, features1.data(), 0.8f);

    // Same sound - should be familiar
    auto signal2 = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features2(config.feature_dim);
    audio_cortex_process(cortex, signal2.data(), config.frame_size, 1, features2.data());

    float novelty2 = audio_cortex_compute_novelty(cortex, features2.data());
    EXPECT_LT(novelty2, 0.5f);  // Should be familiar

    // Different sound - should be more novel
    auto signal3 = generate_sine_wave(1000.0f, config.frame_size);
    std::vector<float> features3(config.feature_dim);
    audio_cortex_process(cortex, signal3.data(), config.frame_size, 1, features3.data());

    float novelty3 = audio_cortex_compute_novelty(cortex, features3.data());
    EXPECT_GT(novelty3, novelty2);  // More novel than familiar sound
}

TEST_F(AudioCortexTest, MemoryConsolidation) {
    auto signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features(config.feature_dim);
    audio_cortex_process(cortex, signal.data(), config.frame_size, 1, features.data());

    bool success = audio_cortex_consolidate_memory(cortex, features.data(), 0.9f, "test_sound");
    EXPECT_TRUE(success);

    audio_cortex_stats_t stats;
    audio_cortex_get_stats(cortex, &stats);
    EXPECT_EQ(stats.memories_stored, 1);
}

//=============================================================================
// Temporal Processing Tests
//=============================================================================

TEST_F(AudioCortexTest, OnsetDetection) {
    // Generate signal with onset (silence -> sound)
    std::vector<float> signal(config.frame_size, 0.0f);

    bool onset, offset;

    // First frame: silence
    audio_cortex_detect_temporal_events(cortex, signal.data(), config.frame_size, &onset, &offset);
    EXPECT_FALSE(onset);

    // Second frame: loud sound (onset)
    auto loud_signal = generate_sine_wave(440.0f, config.frame_size, 1.0f);
    audio_cortex_detect_temporal_events(cortex, loud_signal.data(), config.frame_size, &onset, &offset);
    EXPECT_TRUE(onset);  // Detected onset
}

TEST_F(AudioCortexTest, EnvelopeExtraction) {
    auto signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> envelope(config.frame_size);

    bool success = audio_cortex_compute_envelope(cortex, signal.data(), config.frame_size, envelope.data());
    EXPECT_TRUE(success);

    // Envelope should be non-negative
    for (uint32_t i = 0; i < config.frame_size; i++) {
        EXPECT_GE(envelope[i], 0.0f);
    }

    // Envelope should be smooth
    for (uint32_t i = 1; i < config.frame_size - 1; i++) {
        float diff = fabsf(envelope[i] - envelope[i-1]);
        EXPECT_LT(diff, 0.5f);  // No sudden jumps
    }
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(AudioCortexTest, Statistics) {
    audio_cortex_stats_t stats;
    bool success = audio_cortex_get_stats(cortex, &stats);
    EXPECT_TRUE(success);
    EXPECT_EQ(stats.frames_processed, 0);
    EXPECT_EQ(stats.memories_stored, 0);

    // Process a frame
    auto signal = generate_sine_wave(440.0f, config.frame_size);
    std::vector<float> features(config.feature_dim);
    audio_cortex_process(cortex, signal.data(), config.frame_size, 1, features.data());

    success = audio_cortex_get_stats(cortex, &stats);
    EXPECT_TRUE(success);
    EXPECT_EQ(stats.frames_processed, 1);
}

//=============================================================================
// Stereo Processing Tests
//=============================================================================

TEST_F(AudioCortexTest, StereoProcessing) {
    // Create stereo cortex
    audio_cortex_config_t stereo_config = config;
    stereo_config.num_channels = 2;
    audio_cortex_t* stereo_cortex = audio_cortex_create(&stereo_config);
    ASSERT_NE(stereo_cortex, nullptr);

    // Generate stereo signal (interleaved)
    std::vector<float> stereo_signal(config.frame_size * 2);
    for (uint32_t i = 0; i < config.frame_size; i++) {
        float t = (float)i / config.sample_rate;
        stereo_signal[i * 2] = sinf(2.0f * M_PI * 440.0f * t);      // Left
        stereo_signal[i * 2 + 1] = sinf(2.0f * M_PI * 880.0f * t);  // Right
    }

    std::vector<float> features(config.feature_dim);
    bool success = audio_cortex_process(
        stereo_cortex,
        stereo_signal.data(),
        config.frame_size,
        2,
        features.data()
    );
    EXPECT_TRUE(success);

    audio_cortex_destroy(stereo_cortex);
}
