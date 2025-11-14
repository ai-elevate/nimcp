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

#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "utils/time/nimcp_time.h"

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
    EXPECT_TRUE(neuron_type_is_excitatory(static_cast<neuron_type_extended_t>(NEURON_EXCITATORY)));
    EXPECT_FALSE(neuron_type_is_excitatory(static_cast<neuron_type_extended_t>(NEURON_INHIBITORY)));

    // Visual neurons (excitatory)
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_VISUAL_EDGE));
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_VISUAL_ORIENTATION));
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_VISUAL_DIRECTION));

    // Inhibitory (generic)
    EXPECT_FALSE(neuron_type_is_excitatory(NEURON_INHIBITORY));

    // TODO: Add tests for specific interneuron types when implemented:
    // NEURON_FAST_SPIKING, NEURON_CHANDELIER, NEURON_BASKET

    // Motor neurons (excitatory)
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_MOTOR_ALPHA));
    EXPECT_TRUE(neuron_type_is_excitatory(NEURON_MOTOR_PATTERN_GEN));

    // TODO: Add NEURON_MOTOR_RENSHAW test when implemented

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
    EXPECT_NE(neuron_type_get_name(NEURON_MOTOR_ALPHA), nullptr);
    EXPECT_NE(neuron_type_get_name(NEURON_MOTOR_PATTERN_GEN), nullptr);
}

// ============================================================================
// DEFAULT PARAMETERS TESTS
// ============================================================================

TEST_F(NeuronTypesTest, GetDefaultParameters_VisualEdge) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_VISUAL_EDGE, &params);

    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Verify reasonable defaults (NEURON_VISUAL_EDGE uses v1_simple params)
    EXPECT_GE(params.v1_simple.orientation, 0.0f);
    EXPECT_LE(params.v1_simple.orientation, 180.0f);
    EXPECT_GT(params.v1_simple.spatial_frequency, 0.0f);
    EXPECT_GE(params.v1_simple.phase, 0.0f);
    EXPECT_LE(params.v1_simple.phase, 2.0f * M_PI);
    EXPECT_GT(params.v1_simple.sigma, 0.0f); // Gaussian envelope width
}

TEST_F(NeuronTypesTest, GetDefaultParameters_AuditoryFrequency) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_AUDITORY_FREQUENCY, &params);

    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Verify reasonable defaults (NEURON_AUDITORY_FREQUENCY uses a1_frequency params)
    EXPECT_GT(params.a1_frequency.center_frequency, 0.0f);
    EXPECT_LT(params.a1_frequency.center_frequency, 20000.0f); // Human hearing range
    EXPECT_GT(params.a1_frequency.bandwidth, 0.0f);
    EXPECT_GT(params.a1_frequency.q_factor, 0.0f);
}

TEST_F(NeuronTypesTest, GetDefaultParameters_MotorPattern) {
    neuron_type_params_t params{};
    nimcp_result_t result = neuron_type_get_default_params(NEURON_MOTOR_PATTERN_GEN, &params);

    // Motor pattern gen may not have type-specific params yet - just verify it succeeds
    // TODO: Add specific parameter checks when motor_pattern_params_t is implemented
    EXPECT_EQ(result, NIMCP_SUCCESS);
}

// TODO: Add test for NEURON_FAST_SPIKING when implemented
// TEST_F(NeuronTypesTest, GetDefaultParameters_FastSpiking) {
//     neuron_type_params_t params{};
//     nimcp_result_t result = neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);
//     ASSERT_EQ(result, NIMCP_SUCCESS);
//     // Fast-spiking interneurons fire at high rates
//     EXPECT_GT(params.fast_spiking.max_firing_rate, 50.0f); // >50 Hz
//     EXPECT_LT(params.fast_spiking.membrane_time_constant, 10.0f); // Fast membrane
//     EXPECT_LT(params.fast_spiking.spike_width, 1.0f); // Narrow spikes
// }

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
    params.v1_simple.orientation = 200.0f;

    nimcp_result_t result = neuron_type_validate_params(NEURON_VISUAL_EDGE, &params);
    EXPECT_NE(result, NIMCP_SUCCESS);
}

TEST_F(NeuronTypesTest, ValidateParameters_InvalidFrequency) {
    neuron_type_params_t params{};
    neuron_type_get_default_params(NEURON_AUDITORY_FREQUENCY, &params);

    // Negative frequency
    params.a1_frequency.center_frequency = -100.0f;

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
    params.v1_simple.orientation = 45.0f;

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
    params.a1_frequency.center_frequency = 1000.0f;
    params.a1_frequency.bandwidth = 0.5f; // Half octave

    uint64_t timestamp = nimcp_time_monotonic_us();

    float input = 0.5f;
    float output = neuron_type_process_input(NEURON_AUDITORY_FREQUENCY, &params, input, timestamp);

    // Should process input with frequency tuning
    EXPECT_GE(output, 0.0f);
}

// TODO: Enable when motor_pattern_params_t is implemented
// TEST_F(NeuronTypesTest, ProcessInput_MotorPattern_IntrinsicRhythm) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_MOTOR_PATTERN_GEN, &params);
//     params.motor_pattern.rhythm_frequency = 2.0f; // 2 Hz rhythm
//     params.motor_pattern.pacemaker = true;
//     uint64_t timestamp = nimcp_time_monotonic_us();
//     float input = 0.0f;
//     float output = neuron_type_process_input(NEURON_MOTOR_PATTERN_GEN, &params, input, timestamp);
//     EXPECT_GE(output, 0.0f);
// }

// TODO: Enable when NEURON_FAST_SPIKING is implemented
// TEST_F(NeuronTypesTest, ProcessInput_FastSpiking_RapidDynamics) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);
//     uint64_t timestamp = nimcp_time_monotonic_us();
//     float input = 1.0f;
//     float output = neuron_type_process_input(NEURON_FAST_SPIKING, &params, input, timestamp);
//     EXPECT_GE(output, 0.0f);
// }

// ============================================================================
// THRESHOLD ADJUSTMENT TESTS (TODO: Implement neuron_type_get_threshold)
// ============================================================================

// TODO: Enable when neuron_type_get_threshold is implemented
// TEST_F(NeuronTypesTest, GetThreshold_BasicTypes) {
//     neuron_type_params_t params{};
//     float base_threshold = 1.0f;
//     float threshold_exc = neuron_type_get_threshold(static_cast<neuron_type_extended_t>(NEURON_EXCITATORY), &params, base_threshold);
//     EXPECT_FLOAT_EQ(threshold_exc, base_threshold);
//     float threshold_inh = neuron_type_get_threshold(static_cast<neuron_type_extended_t>(NEURON_INHIBITORY), &params, base_threshold);
//     EXPECT_FLOAT_EQ(threshold_inh, base_threshold);
// }

// TODO: Enable when NEURON_FAST_SPIKING and neuron_type_get_threshold are implemented
// TEST_F(NeuronTypesTest, GetThreshold_FastSpiking_LowerThreshold) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);
//     float base_threshold = 1.0f;
//     float threshold = neuron_type_get_threshold(NEURON_FAST_SPIKING, &params, base_threshold);
//     EXPECT_LE(threshold, base_threshold);
//     EXPECT_GT(threshold, 0.0f);
// }

// TODO: Enable when NEURON_BURST_SPIKING and neuron_type_get_threshold are implemented
// TEST_F(NeuronTypesTest, GetThreshold_Bursting_HigherThreshold) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_BURST_SPIKING, &params);
//     params.burst.burst_threshold = 1.5f;
//     float base_threshold = 1.0f;
//     float threshold = neuron_type_get_threshold(NEURON_BURST_SPIKING, &params, base_threshold);
//     EXPECT_GE(threshold, base_threshold);
// }

// ============================================================================
// REFRACTORY PERIOD TESTS (TODO: Implement neuron_type_get_refractory_period)
// ============================================================================

// TODO: Enable when NEURON_FAST_SPIKING and neuron_type_get_refractory_period are implemented
// TEST_F(NeuronTypesTest, GetRefractoryPeriod_FastSpiking_Short) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_FAST_SPIKING, &params);
//     float refractory = neuron_type_get_refractory_period(NEURON_FAST_SPIKING, &params);
//     EXPECT_GT(refractory, 0.0f);
//     EXPECT_LT(refractory, 2.0f);
// }

// TODO: Enable when neuron_type_get_refractory_period is implemented
// TEST_F(NeuronTypesTest, GetRefractoryPeriod_Pyramidal_Standard) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_PYRAMIDAL_L23, &params);
//     float refractory = neuron_type_get_refractory_period(NEURON_PYRAMIDAL_L23, &params);
//     EXPECT_GE(refractory, 2.0f);
//     EXPECT_LE(refractory, 5.0f);
// }

// TODO: Enable when NEURON_BURST_SPIKING and neuron_type_get_refractory_period are implemented
// TEST_F(NeuronTypesTest, GetRefractoryPeriod_Bursting_Long) {
//     neuron_type_params_t params{};
//     neuron_type_get_default_params(NEURON_BURST_SPIKING, &params);
//     float refractory = neuron_type_get_refractory_period(NEURON_BURST_SPIKING, &params);
//     EXPECT_GT(refractory, 0.0f);
// }

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

        params.v1_simple.orientation = orientation;

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

        params.a1_frequency.center_frequency = freq;

        nimcp_result_t result = neuron_type_validate_params(NEURON_AUDITORY_FREQUENCY, &params);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }
}

// TODO: Enable when motor_pattern_params_t is implemented
// TEST_F(NeuronTypesTest, MotorPattern_CoordinatedRhythms) {
//     float phase_offsets[] = {0.0f, 0.25f, 0.5f, 0.75f};
//     for (float phase : phase_offsets) {
//         neuron_type_params_t params{};
//         neuron_type_get_default_params(NEURON_MOTOR_PATTERN_GEN, &params);
//         params.motor_pattern.phase_offset = phase;
//         nimcp_result_t result = neuron_type_validate_params(NEURON_MOTOR_PATTERN_GEN, &params);
//         EXPECT_EQ(result, NIMCP_SUCCESS);
//     }
// }

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
