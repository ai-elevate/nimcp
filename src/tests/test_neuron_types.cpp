/**
 * @file test_neuron_types.cpp
 * @brief TDD tests for neuron type specialization system
 *
 * Test Coverage:
 * - Type classification (E/I mapping)
 * - Default parameter generation
 * - Parameter validation
 * - Type-specific input processing
 * - Threshold adjustments
 * - Refractory period customization
 * - Visual neuron types (edge, orientation)
 * - Auditory neuron types (frequency, onset)
 * - Motor neuron types (pattern generators)
 * - Interneuron types (fast-spiking, bursting)
 */

#include <gtest/gtest.h>

extern "C" {
#include "nimcp_neuron_types.h"
#include "utils/nimcp_time.h"
}

#include <cmath>

// ============================================================================
// TEST FIXTURE
// ============================================================================

class NeuronTypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize any common test data
    }

    void TearDown() override {
        // Cleanup
    }
};

// ============================================================================
// TYPE CLASSIFICATION TESTS
// ============================================================================

TEST_F(NeuronTypesTest, ExcitatoryInhibitoryClassification) {
    // Basic types
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_EXCITATORY));
    EXPECT_FALSE(neuron_type_is_excitatory(NEURON_INHIBITORY));

    // Visual neurons (excitatory)
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_VISUAL_EDGE));
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_VISUAL_ORIENTATION));
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_VISUAL_DIRECTION));

    // Interneurons (inhibitory)
    EXPECT_FALSE(neuron_type_is_excitatory(NEURON_FAST_SPIKING));
    EXPECT_FALSE(neuron_type_is_excitatory(NEURON_CHANDELIER));
    EXPECT_FALSE(neuron_type_is_excitatory(NEURON_BASKET));

    // Motor neurons (mixed)
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_MOTOR_ALPHA));
    EXPECT_FALSE(neuron_type_is_excitatory(NEURON_MOTOR_RENSHAW)); // Inhibitory

    // Pyramidal neurons (excitatory)
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_PYRAMIDAL_L23));
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_PYRAMIDAL_L5_THICK));
}

TEST_F(NeuronTypesTest, TypeNameRetrieval) {
    const char* name = neuron_type_get_name(NEURON_VISUAL_EDGE);
    ASSERT_NE(name, nullptr);
    EXPECT_STRNE(name, "");

    // Check a few specific types
    EXPECT_NE(neuron_type_get_name(NEURON_AUDITORY_FREQUENCY), nullptr);
    EXPECT_NE(neuron_type_get_name(NEURON_FAST_SPIKING), nullptr);
    EXPECT_NE(neuron_type_get_name(NEURON_MOTOR_PATTERN_GEN), nullptr);
}

// ============================================================================
// DEFAULT PARAMETERS TESTS
// ============================================================================

TEST_F(NeuronTypesTest, GetDefaultParameters_VisualEdge) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Verify reasonable defaults
    EXPECT_GE(params.visual_edge.orientation, 0.0f);
    EXPECT_LE(params.visual_edge.orientation, 180.0f);
    EXPECT_GT(params.visual_edge.spatial_frequency, 0.0f);
    EXPECT_GE(params.visual_edge.phase, 0.0f);
    EXPECT_LE(params.visual_edge.phase, 2.0f * M_PI);
    EXPECT_GT(params.visual_edge.receptive_field_size, 0.0f);
}

TEST_F(NeuronTypesTest, GetDefaultParameters_AuditoryFrequency) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_AUDITORY_FREQUENCY, &params);

    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Verify reasonable defaults
    EXPECT_GT(params.auditory_frequency.center_frequency, 0.0f);
    EXPECT_LT(params.auditory_frequency.center_frequency, 20000.0f); // Human hearing range
    EXPECT_GT(params.auditory_frequency.bandwidth, 0.0f);
    EXPECT_GT(params.auditory_frequency.q_factor, 0.0f);
}

TEST_F(NeuronTypesTest, GetDefaultParameters_MotorPattern) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_MOTOR_PATTERN_GEN, &params);

    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Verify CPG parameters
    EXPECT_GT(params.motor_pattern.rhythm_frequency, 0.0f);
    EXPECT_LT(params.motor_pattern.rhythm_frequency, 100.0f); // Reasonable CPG range
    EXPECT_GT(params.motor_pattern.burst_duration, 0.0f);
    EXPECT_GE(params.motor_pattern.interburst_interval, 0.0f);
}

TEST_F(NeuronTypesTest, GetDefaultParameters_FastSpiking) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);

    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Fast-spiking interneurons fire at high rates
    EXPECT_GT(params.fast_spiking.max_firing_rate, 50.0f); // >50 Hz
    EXPECT_LT(params.fast_spiking.membrane_time_constant, 10.0f); // Fast membrane
    EXPECT_LT(params.fast_spiking.spike_width, 1.0f); // Narrow spikes
}

// ============================================================================
// PARAMETER VALIDATION TESTS
// ============================================================================

TEST_F(NeuronTypesTest, ValidateParameters_ValidVisualEdge) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

    nimcp_result_t result = neuron_type_validate_params(NEURON_VISUAL_EDGE, &params);
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

TEST_F(NeuronTypesTest, ValidateParameters_InvalidOrientation) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

    // Invalid orientation (> 180°)
    params.visual_edge.orientation = 200.0f;

    nimcp_result_t result = neuron_type_validate_params(NEURON_VISUAL_EDGE, &params);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(NeuronTypesTest, ValidateParameters_InvalidFrequency) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_AUDITORY_FREQUENCY, &params);

    // Negative frequency
    params.auditory_frequency.center_frequency = -100.0f;

    nimcp_result_t result = neuron_type_validate_params(NEURON_AUDITORY_FREQUENCY, &params);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

// ============================================================================
// TYPE-SPECIFIC INPUT PROCESSING TESTS
// ============================================================================

TEST_F(NeuronTypesTest, ProcessInput_VisualEdge_OrientationTuning) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

    // Set preferred orientation to 45 degrees
    params.visual_edge.orientation = 45.0f;

    uint64_t timestamp = nimcp_time_monotonic_us();

    // Process input (simulating oriented edge stimulus)
    float input = 1.0f;
    float output = neuron_type_process_input(NEURON_VISUAL_EDGE, &params, input, timestamp);

    // Should produce some output (specific value depends on implementation)
    EXPECT_GE(output, 0.0f);
}

TEST_F(NeuronTypesTest, ProcessInput_AuditoryFrequency_Tuning) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_AUDITORY_FREQUENCY, &params);

    // Set center frequency to 1000 Hz
    params.auditory_frequency.center_frequency = 1000.0f;
    params.auditory_frequency.bandwidth = 0.5f; // Half octave

    uint64_t timestamp = nimcp_time_monotonic_us();

    float input = 0.5f;
    float output = neuron_type_process_input(NEURON_AUDITORY_FREQUENCY, &params, input, timestamp);

    // Should process input with frequency tuning
    EXPECT_GE(output, 0.0f);
}

TEST_F(NeuronTypesTest, ProcessInput_MotorPattern_IntrinsicRhythm) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_MOTOR_PATTERN_GEN, &params);

    params.motor_pattern.rhythm_frequency = 2.0f; // 2 Hz rhythm
    params.motor_pattern.pacemaker = true;

    uint64_t timestamp = nimcp_time_monotonic_us();

    // Even with zero input, pacemaker should produce rhythm
    float input = 0.0f;
    float output = neuron_type_process_input(NEURON_MOTOR_PATTERN_GEN, &params, input, timestamp);

    // Pacemaker neurons can fire autonomously
    EXPECT_GE(output, 0.0f);
}

TEST_F(NeuronTypesTest, ProcessInput_FastSpiking_RapidDynamics) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);

    uint64_t timestamp = nimcp_time_monotonic_us();

    float input = 1.0f;
    float output = neuron_type_process_input(NEURON_FAST_SPIKING, &params, input, timestamp);

    // Fast-spiking neurons respond quickly
    EXPECT_GE(output, 0.0f);
}

// ============================================================================
// THRESHOLD ADJUSTMENT TESTS
// ============================================================================

TEST_F(NeuronTypesTest, GetThreshold_BasicTypes) {
    neuron_type_params_t params{};
    float base_threshold = 1.0f;

    // Basic types should return base threshold unchanged
    float threshold_exc = neuron_type_get_threshold(NEURON_EXCITATORY, &params, base_threshold);
    EXPECT_FLOAT_EQ(threshold_exc, base_threshold);

    float threshold_inh = neuron_type_get_threshold(NEURON_INHIBITORY, &params, base_threshold);
    EXPECT_FLOAT_EQ(threshold_inh, base_threshold);
}

TEST_F(NeuronTypesTest, GetThreshold_FastSpiking_LowerThreshold) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);

    float base_threshold = 1.0f;
    float threshold = neuron_type_get_threshold(NEURON_FAST_SPIKING, &params, base_threshold);

    // Fast-spiking neurons typically have lower thresholds
    EXPECT_LE(threshold, base_threshold);
    EXPECT_GT(threshold, 0.0f);
}

TEST_F(NeuronTypesTest, GetThreshold_Bursting_HigherThreshold) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_BURST_SPIKING, &params);

    params.burst.burst_threshold = 1.5f;

    float base_threshold = 1.0f;
    float threshold = neuron_type_get_threshold(NEURON_BURST_SPIKING, &params, base_threshold);

    // Bursting neurons may have higher threshold for burst initiation
    EXPECT_GE(threshold, base_threshold);
}

// ============================================================================
// REFRACTORY PERIOD TESTS
// ============================================================================

TEST_F(NeuronTypesTest, GetRefractoryPeriod_FastSpiking_Short) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);

    float refractory = neuron_type_get_refractory_period(NEURON_FAST_SPIKING, &params);

    // Fast-spiking neurons have short refractory periods (< 2 ms)
    EXPECT_GT(refractory, 0.0f);
    EXPECT_LT(refractory, 2.0f);
}

TEST_F(NeuronTypesTest, GetRefractoryPeriod_Pyramidal_Standard) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_PYRAMIDAL_L23, &params);

    float refractory = neuron_type_get_refractory_period(NEURON_PYRAMIDAL_L23, &params);

    // Pyramidal neurons have standard refractory (2-5 ms)
    EXPECT_GE(refractory, 2.0f);
    EXPECT_LE(refractory, 5.0f);
}

TEST_F(NeuronTypesTest, GetRefractoryPeriod_Bursting_Long) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_BURST_SPIKING, &params);

    float refractory = neuron_type_get_refractory_period(NEURON_BURST_SPIKING, &params);

    // Bursting neurons may have longer refractory between bursts
    EXPECT_GT(refractory, 0.0f);
}

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

TEST_F(NeuronTypesTest, GetDefaultParameters_NullParams) {
    nimcp_result_t result = neuron_type_get_default_params(NEURON_VISUAL_EDGE, nullptr);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(NeuronTypesTest, ValidateParameters_NullParams) {
    nimcp_result_t result = neuron_type_validate_params(NEURON_VISUAL_EDGE, nullptr);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(NeuronTypesTest, ProcessInput_NullParams) {
    uint64_t timestamp = nimcp_time_monotonic_us();
    float output = neuron_type_process_input(NEURON_VISUAL_EDGE, nullptr, 1.0f, timestamp);

    // Should handle gracefully (return input or 0.0)
    EXPECT_GE(output, 0.0f);
}

// ============================================================================
// COMPLEX NEURON TYPE TESTS
// ============================================================================

TEST_F(NeuronTypesTest, VisualEdge_MultipleOrientations) {
    // Test neurons with different preferred orientations
    for (float orientation = 0.0f; orientation <= 180.0f; orientation += 45.0f) {
        neuron_type_params_t params{};
        neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

        params.visual_edge.orientation = orientation;

        nimcp_result_t result = neuron_type_validate_params(NEURON_VISUAL_EDGE, &params);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }
}

TEST_F(NeuronTypesTest, AuditoryFrequency_LogarithmicSpacing) {
    // Test frequency-tuned neurons across auditory range
    float frequencies[] = {125.0f, 250.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f};

    for (float freq : frequencies) {
        neuron_type_params_t params{};
        neuron_type_get_default_params(NEURON_AUDITORY_FREQUENCY, &params);

        params.auditory_frequency.center_frequency = freq;

        nimcp_result_t result = neuron_type_validate_params(NEURON_AUDITORY_FREQUENCY, &params);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }
}

TEST_F(NeuronTypesTest, MotorPattern_CoordinatedRhythms) {
    // Test CPG neurons with different phase offsets (for locomotion)
    float phase_offsets[] = {0.0f, 0.25f, 0.5f, 0.75f};

    for (float phase : phase_offsets) {
        neuron_type_params_t params{};
        neuron_type_get_default_params(NEURON_MOTOR_PATTERN_GEN, &params);

        params.motor_pattern.phase_offset = phase;

        nimcp_result_t result = neuron_type_validate_params(NEURON_MOTOR_PATTERN_GEN, &params);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

TEST_F(NeuronTypesTest, ProcessInput_Performance) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

    uint64_t start = nimcp_time_monotonic_us();

    // Process 10000 inputs
    const int iterations = 10000;
    float sum = 0.0f;
    for (int i = 0; i < iterations; i++) {
        uint64_t timestamp = start + i * 1000; // 1ms steps
        sum += neuron_type_process_input(NEURON_VISUAL_EDGE, &params, 0.5f, timestamp);
    }

    uint64_t end = nimcp_time_monotonic_us();
    uint64_t duration = end - start;

    // Should process at least 1000 inputs per ms (1M per second)
    float inputs_per_us = (float)iterations / (float)duration;
    EXPECT_GT(inputs_per_us, 1.0f);

    // Use sum to prevent optimization
    EXPECT_GE(sum, 0.0f);
}
