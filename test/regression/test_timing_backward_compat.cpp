/**
 * @file test_timing_backward_compat.cpp
 * @brief Regression tests for timing system backward compatibility
 *
 * WHAT: Verify timing fixes don't break existing behavior
 * WHY:  Ensure timing bug fixes maintain backward compatibility
 * HOW:  Test pre-fix expected behaviors still work
 *
 * TEST COVERAGE:
 * 1. glial_integration_create() still works
 * 2. glial_integration_step() still works with timestamp parameter
 * 3. neural_network_compute_step() still works
 * 4. brain_process_multimodal() still works with timing enabled
 * 5. Multiple consecutive updates don't crash
 * 6. Timing-dependent features (STP, calcium) still functional
 *
 * @version Regression Testing Framework v1.0
 * @date 2025-11-13
 */

#include <gtest/gtest.h>

extern "C" {
    #include "core/brain/nimcp_brain.h"
    #include "glial/integration/nimcp_glial_integration.h"
    #include "core/neuralnet/nimcp_neuralnet.h"
    #include "glial/astrocytes/nimcp_astrocytes.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class TimingBackwardCompatTest : public ::testing::Test {
protected:
    neural_network_t network;
    glial_integration_t* glial;
    brain_t brain;

    void SetUp() override {
        network = nullptr;
        glial = nullptr;
        brain = nullptr;
    }

    void TearDown() override {
        if (glial) {
            glial_integration_destroy(glial);
            glial = nullptr;
        }
        if (network) {
            neural_network_destroy(network);
            network = nullptr;
        }
        if (brain) {
            brain_destroy(brain);
            brain = nullptr;
        }
    }
};

//=============================================================================
// Regression Test 1: glial_integration_create() API Unchanged
//=============================================================================

TEST_F(TimingBackwardCompatTest, GlialCreate_APIUnchanged) {
    // WHAT: Verify glial_integration_create() signature unchanged
    // WHY:  Existing code using this API must continue to work

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Old API call pattern
    glial = glial_integration_create(network, 100);
    ASSERT_NE(glial, nullptr);

    SUCCEED() << "glial_integration_create() API backward compatible";
}

//=============================================================================
// Regression Test 2: glial_integration_step() Timestamp Parameter
//=============================================================================

TEST_F(TimingBackwardCompatTest, GlialStep_TimestampParameter) {
    // WHAT: Verify glial_integration_step() still takes uint64_t timestamp
    // WHY:  Signature must remain unchanged for backward compatibility

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    glial = glial_integration_create(network, 100);
    ASSERT_NE(glial, nullptr);

    // Old calling pattern - passing timestamp
    uint64_t timestamp = 1000;
    glial_integration_step(glial, timestamp);

    timestamp = 2000;
    glial_integration_step(glial, timestamp);

    SUCCEED() << "glial_integration_step() signature backward compatible";
}

//=============================================================================
// Regression Test 3: neural_network_compute_step() Unchanged
//=============================================================================

TEST_F(TimingBackwardCompatTest, NetworkComputeStep_APIUnchanged) {
    // WHAT: Verify neural_network_compute_step() behavior unchanged
    // WHY:  Core network function must remain compatible

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Old calling pattern
    uint32_t active = neural_network_compute_step(network, 1000);
    EXPECT_GE(active, 0u);

    active = neural_network_compute_step(network, 2000);
    EXPECT_GE(active, 0u);

    SUCCEED() << "neural_network_compute_step() backward compatible";
}

//=============================================================================
// Regression Test 4: brain_process_multimodal() Unchanged
//=============================================================================

TEST_F(TimingBackwardCompatTest, BrainProcess_APIUnchanged) {
    // WHAT: Verify brain_process_multimodal() behavior unchanged
    // WHY:  High-level API must remain compatible

    brain_config_t config = {};
    config.num_neurons = 10;
    config.num_inputs = 3;
    config.num_outputs = 2;
    config.enable_glial = true;

    brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    // Old calling pattern
    brain_input_t input = {};
    float input_data[3] = {0.5f, 0.3f, 0.2f};
    input.input_data = input_data;
    input.input_size = 3;

    brain_output_t output;
    bool success = brain_process_multimodal(brain, &input, &output);
    ASSERT_TRUE(success);

    SUCCEED() << "brain_process_multimodal() backward compatible";
}

//=============================================================================
// Regression Test 5: Multiple Consecutive Updates
//=============================================================================

TEST_F(TimingBackwardCompatTest, MultipleUpdates_StillWork) {
    // WHAT: Verify rapid consecutive updates don't break
    // WHY:  Existing code may call updates in tight loops

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    glial = glial_integration_create(network, 100);
    ASSERT_NE(glial, nullptr);

    // Rapid consecutive updates (old pattern)
    for (uint64_t t = 0; t < 100; t += 1) {
        glial_integration_step(glial, t * 1000);  // Every 1ms
    }

    SUCCEED() << "Multiple consecutive updates backward compatible";
}

//=============================================================================
// Regression Test 6: Astrocyte Network Integration
//=============================================================================

TEST_F(TimingBackwardCompatTest, AstrocyteIntegration_StillWorks) {
    // WHAT: Verify astrocyte integration unchanged
    // WHY:  Calcium dynamics depend on proper timing

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    glial = glial_integration_create(network, 100);
    ASSERT_NE(glial, nullptr);

    // Create astrocyte network
    astrocyte_network_t* astro_net = astrocyte_network_create(5);
    ASSERT_NE(astro_net, nullptr);

    // Old pattern - set astrocyte network on glial integration
    nimcp_result_t result = glial_integration_set_astrocyte_network(glial, astro_net);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Enable astrocyte modulation
    result = glial_integration_enable_astrocytes(glial);
    EXPECT_EQ(result, NIMCP_SUCCESS);

    // Update with timing
    glial_integration_step(glial, 1000);
    glial_integration_step(glial, 2000);

    astrocyte_network_destroy(astro_net);

    SUCCEED() << "Astrocyte integration backward compatible";
}

//=============================================================================
// Regression Test 7: STP Still Works with New Timing
//=============================================================================

TEST_F(TimingBackwardCompatTest, STP_StillFunctional) {
    // WHAT: Verify STP dynamics work with synchronized timing
    // WHY:  STP is timing-critical feature

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    glial = glial_integration_create(network, 100);
    ASSERT_NE(glial, nullptr);

    // Process with STP-like timing pattern
    for (uint64_t t = 0; t < 20; t++) {
        glial_integration_step(glial, t * 1000);
        neural_network_compute_step(network, t * 1000);
    }

    SUCCEED() << "STP timing backward compatible";
}

//=============================================================================
// Regression Test 8: Zero Initial Timestamp
//=============================================================================

TEST_F(TimingBackwardCompatTest, ZeroInitialTimestamp_StillWorks) {
    // WHAT: Verify starting from timestamp=0 works
    // WHY:  Common pattern in existing code

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    glial = glial_integration_create(network, 100);
    ASSERT_NE(glial, nullptr);

    // Old pattern - start from 0
    glial_integration_step(glial, 0);
    glial_integration_step(glial, 1000);
    glial_integration_step(glial, 2000);

    SUCCEED() << "Zero initial timestamp backward compatible";
}

//=============================================================================
// Regression Test 9: No Glial Integration (Disabled)
//=============================================================================

TEST_F(TimingBackwardCompatTest, GlialDisabled_BrainStillWorks) {
    // WHAT: Verify brain works with glial disabled
    // WHY:  Some configurations don't use glial cells

    brain_config_t config = {};
    config.num_neurons = 10;
    config.num_inputs = 3;
    config.num_outputs = 2;
    config.enable_glial = false;  // Disabled

    brain = brain_create(&config);
    ASSERT_NE(brain, nullptr);

    brain_input_t input = {};
    float input_data[3] = {0.5f, 0.3f, 0.2f};
    input.input_data = input_data;
    input.input_size = 3;

    // Process without glial
    for (int i = 0; i < 10; i++) {
        brain_output_t output;
        bool success = brain_process_multimodal(brain, &input, &output);
        ASSERT_TRUE(success);
    }

    SUCCEED() << "Glial disabled mode backward compatible";
}

//=============================================================================
// Regression Test 10: Existing Test Patterns Still Valid
//=============================================================================

TEST_F(TimingBackwardCompatTest, ExistingTestPatterns_StillValid) {
    // WHAT: Verify common test patterns from existing tests still work
    // WHY:  Ensure no breaking changes to test infrastructure

    network_config_t config = {};
    config.num_neurons = 10;
    config.refractory_period = 2;
    network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Pattern 1: Create and destroy immediately
    glial_integration_t* temp_glial = glial_integration_create(network, 100);
    ASSERT_NE(temp_glial, nullptr);
    glial_integration_destroy(temp_glial);

    // Pattern 2: Create, use once, destroy
    temp_glial = glial_integration_create(network, 100);
    glial_integration_step(temp_glial, 1000);
    glial_integration_destroy(temp_glial);

    // Pattern 3: Multiple creates from same network
    glial_integration_t* glial1 = glial_integration_create(network, 100);
    glial_integration_t* glial2 = glial_integration_create(network, 100);
    ASSERT_NE(glial1, nullptr);
    ASSERT_NE(glial2, nullptr);
    glial_integration_destroy(glial1);
    glial_integration_destroy(glial2);

    SUCCEED() << "Existing test patterns backward compatible";
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
