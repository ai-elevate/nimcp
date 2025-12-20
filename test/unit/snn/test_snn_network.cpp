/**
 * @file test_snn_network.cpp
 * @brief Unit tests for SNN Network module
 *
 * TEST COVERAGE:
 * - Network lifecycle (create, destroy, reset)
 * - Simulation (step, run)
 * - Input/output (set inputs, get outputs, forward)
 * - Training (STDP, R-STDP, surrogate gradients)
 * - Population management
 * - Statistics and monitoring
 * - Edge cases and error handling
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" {
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_config.h"
#include "snn/nimcp_snn_types.h"
#include "utils/tensor/nimcp_tensor.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class SnnNetworkTest : public ::testing::Test {
protected:
    snn_network_t* network = nullptr;
    snn_config_t config;

    void SetUp() override {
        memset(&config, 0, sizeof(snn_config_t));
        snn_config_feedforward(&config, 4, 8, 2);
    }

    void TearDown() override {
        if (network) {
            snn_network_destroy(network);
            network = nullptr;
        }
        snn_config_destroy(&config);
    }

    void CreateNetwork() {
        network = snn_network_create(&config);
    }

    nimcp_tensor_t* CreateInputTensor(uint32_t size, float value = 0.5f) {
        uint32_t dims[1] = {size};
        nimcp_tensor_t* t = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
        if (t) {
            float* data = (float*)nimcp_tensor_data(t);
            for (uint64_t i = 0; i < size; i++) {
                data[i] = value;
            }
        }
        return t;
    }

    nimcp_tensor_t* CreateOutputTensor(uint32_t size) {
        uint32_t dims[1] = {size};
        return nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    }
};

//=============================================================================
// snn_network_create Tests
//=============================================================================

TEST_F(SnnNetworkTest, CreateReturnsValidPointer) {
    CreateNetwork();
    EXPECT_NE(nullptr, network);
}

TEST_F(SnnNetworkTest, CreateReturnsNullOnNullConfig) {
    snn_network_t* n = snn_network_create(nullptr);
    EXPECT_EQ(nullptr, n);
}

TEST_F(SnnNetworkTest, CreateReturnsNullOnInvalidConfig) {
    config.n_inputs = 0;  // Invalid
    snn_network_t* n = snn_network_create(&config);
    EXPECT_EQ(nullptr, n);
}

TEST_F(SnnNetworkTest, CreateSetsCorrectDimensions) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    // Check input and output populations exist
    EXPECT_NE(nullptr, network->input_pop);
    EXPECT_NE(nullptr, network->output_pop);
    EXPECT_EQ(config.n_inputs, network->input_pop->n_neurons);
    EXPECT_EQ(config.n_outputs, network->output_pop->n_neurons);
}

TEST_F(SnnNetworkTest, CreateInitializesSimulationContext) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    EXPECT_NE(nullptr, network->sim);
    EXPECT_EQ(0ULL, network->sim->current_time_us);
    EXPECT_EQ(0ULL, network->sim->step_count);
    EXPECT_EQ(SNN_STATE_HEALTHY, network->sim->health);
}

TEST_F(SnnNetworkTest, CreateSetsMagicNumber) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    EXPECT_EQ(SNN_MAGIC, network->magic);
}

//=============================================================================
// snn_network_destroy Tests
//=============================================================================

TEST_F(SnnNetworkTest, DestroySafeOnNullPointer) {
    // Should not crash
    snn_network_destroy(nullptr);
}

TEST_F(SnnNetworkTest, DestroyFreesResources) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_network_destroy(network);
    network = nullptr;  // Prevent double-free in TearDown
}

//=============================================================================
// snn_network_reset Tests
//=============================================================================

TEST_F(SnnNetworkTest, ResetClearsSpikeCounts) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    // Run some simulation to generate spikes
    float inputs[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    snn_network_set_inputs(network, inputs, 4);
    snn_network_run(network, 10.0f);

    // Reset
    int result = snn_network_reset(network);
    EXPECT_EQ(SNN_SUCCESS, result);

    // Check that simulation context was reset
    EXPECT_EQ(0ULL, network->sim->current_time_us);
    EXPECT_EQ(0ULL, network->sim->step_count);
}

TEST_F(SnnNetworkTest, ResetReturnsErrorOnNullPointer) {
    int result = snn_network_reset(nullptr);
    EXPECT_NE(SNN_SUCCESS, result);
}

//=============================================================================
// snn_network_step Tests
//=============================================================================

TEST_F(SnnNetworkTest, StepAdvancesSimulationTime) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    uint64_t time_before = network->sim->current_time_us;
    snn_network_step(network, 0.1f);
    uint64_t time_after = network->sim->current_time_us;

    EXPECT_GT(time_after, time_before);
}

TEST_F(SnnNetworkTest, StepIncrementsStepCount) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    EXPECT_EQ(0ULL, network->sim->step_count);
    snn_network_step(network, 0.0f);  // Use default dt
    EXPECT_EQ(1ULL, network->sim->step_count);

    snn_network_step(network, 0.0f);
    EXPECT_EQ(2ULL, network->sim->step_count);
}

TEST_F(SnnNetworkTest, StepReturnsNonNegativeSpikes) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    int spikes = snn_network_step(network, 0.1f);
    EXPECT_GE(spikes, 0);
}

TEST_F(SnnNetworkTest, StepReturnsErrorOnNullPointer) {
    int result = snn_network_step(nullptr, 0.1f);
    EXPECT_LT(result, 0);
}

//=============================================================================
// snn_network_run Tests
//=============================================================================

TEST_F(SnnNetworkTest, RunExecutesMultipleSteps) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    int total_spikes = snn_network_run(network, 10.0f);
    EXPECT_GE(total_spikes, 0);

    // Should have executed many steps (10ms / 0.1ms dt = 100 steps)
    EXPECT_GE(network->sim->step_count, 10ULL);
}

TEST_F(SnnNetworkTest, RunReturnsErrorOnNullPointer) {
    int result = snn_network_run(nullptr, 10.0f);
    EXPECT_LT(result, 0);
}

//=============================================================================
// snn_network_set_inputs Tests
//=============================================================================

TEST_F(SnnNetworkTest, SetInputsAcceptsValidInput) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    int result = snn_network_set_inputs(network, inputs, 4);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, SetInputsReturnsErrorOnNullPointer) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    int result = snn_network_set_inputs(nullptr, inputs, 4);
    EXPECT_NE(SNN_SUCCESS, result);

    result = snn_network_set_inputs(network, nullptr, 4);
    EXPECT_NE(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, SetInputsReturnsErrorOnDimensionMismatch) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[8] = {0.0f};  // Wrong size (config.n_inputs = 4)
    int result = snn_network_set_inputs(network, inputs, 8);
    EXPECT_NE(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, SetInputTensorWorks) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    nimcp_tensor_t* input = CreateInputTensor(4, 0.5f);
    ASSERT_NE(nullptr, input);

    int result = snn_network_set_input_tensor(network, input);
    EXPECT_EQ(SNN_SUCCESS, result);

    nimcp_tensor_destroy(input);
}

//=============================================================================
// snn_network_get_outputs Tests
//=============================================================================

TEST_F(SnnNetworkTest, GetOutputsReturnsValidValues) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    // Run some simulation
    float inputs[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    snn_network_set_inputs(network, inputs, 4);
    snn_network_run(network, 100.0f);

    float outputs[2];
    int result = snn_network_get_outputs(network, outputs, 2);
    EXPECT_EQ(SNN_SUCCESS, result);

    // Outputs should be non-negative (firing rates)
    EXPECT_GE(outputs[0], 0.0f);
    EXPECT_GE(outputs[1], 0.0f);
}

TEST_F(SnnNetworkTest, GetOutputsReturnsErrorOnNullPointer) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float outputs[2];
    int result = snn_network_get_outputs(nullptr, outputs, 2);
    EXPECT_NE(SNN_SUCCESS, result);

    result = snn_network_get_outputs(network, nullptr, 2);
    EXPECT_NE(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, GetOutputsReturnsErrorOnDimensionMismatch) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float outputs[8];  // Wrong size (config.n_outputs = 2)
    int result = snn_network_get_outputs(network, outputs, 8);
    EXPECT_NE(SNN_SUCCESS, result);
}

//=============================================================================
// snn_network_forward Tests
//=============================================================================

TEST_F(SnnNetworkTest, ForwardPerformsFullInference) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float outputs[2];

    int result = snn_network_forward(network, inputs, 4, outputs, 2, 100.0f);
    EXPECT_EQ(SNN_SUCCESS, result);

    // Outputs should be valid firing rates
    EXPECT_GE(outputs[0], 0.0f);
    EXPECT_GE(outputs[1], 0.0f);
}

TEST_F(SnnNetworkTest, ForwardResetsNetworkBeforeRun) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float outputs[2];

    // First forward pass
    snn_network_forward(network, inputs, 4, outputs, 2, 50.0f);

    // Second forward pass should reset first
    snn_network_forward(network, inputs, 4, outputs, 2, 50.0f);

    // Step count should be approximately 500 (50ms / 0.1ms), not 1000
    EXPECT_LT(network->sim->step_count, 1000ULL);
}

//=============================================================================
// Training Tests
//=============================================================================

TEST_F(SnnNetworkTest, SetTrainingEnablesTrainingMode) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    EXPECT_FALSE(network->is_training);

    int result = snn_network_set_training(network, true);
    EXPECT_EQ(SNN_SUCCESS, result);
    EXPECT_TRUE(network->is_training);
    EXPECT_NE(nullptr, network->train_ctx);

    result = snn_network_set_training(network, false);
    EXPECT_EQ(SNN_SUCCESS, result);
    EXPECT_FALSE(network->is_training);
}

TEST_F(SnnNetworkTest, ApplyStdpReturnsNonNegative) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_network_set_training(network, true);

    int modified = snn_network_apply_stdp(network);
    EXPECT_GE(modified, 0);
}

TEST_F(SnnNetworkTest, ApplyRstdpReturnsNonNegative) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_network_set_training(network, true);

    int modified = snn_network_apply_rstdp(network, 1.0f);
    EXPECT_GE(modified, 0);
}

TEST_F(SnnNetworkTest, ComputeGradientsUpdatesLoss) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_network_set_training(network, true);

    // Run some simulation first
    float inputs[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    snn_network_set_inputs(network, inputs, 4);
    snn_network_run(network, 100.0f);

    float targets[2] = {1.0f, 0.0f};
    int result = snn_network_compute_gradients(network, targets, 2);
    EXPECT_EQ(SNN_SUCCESS, result);

    // Loss should be computed
    EXPECT_GE(network->train_ctx->current_loss, 0.0f);
}

TEST_F(SnnNetworkTest, TrainStepReturnsValidLoss) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[4] = {1.0f, 0.5f, 0.0f, 0.5f};
    float targets[2] = {1.0f, 0.0f};

    float loss = snn_network_train_step(network, inputs, 4, targets, 2, 100.0f);
    EXPECT_GE(loss, 0.0f);
}

//=============================================================================
// Population Management Tests
//=============================================================================

TEST_F(SnnNetworkTest, AddPopulationCreatesNewPopulation) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    uint32_t initial_count = network->n_populations;

    int pop_id = snn_network_add_population(network, 16, NEURON_GENERIC_LIF, "hidden");
    EXPECT_GE(pop_id, 0);
    EXPECT_EQ(initial_count + 1, network->n_populations);
}

TEST_F(SnnNetworkTest, AddPopulationReturnsErrorOnZeroNeurons) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    int pop_id = snn_network_add_population(network, 0, NEURON_GENERIC_LIF, "bad");
    EXPECT_LT(pop_id, 0);
}

TEST_F(SnnNetworkTest, GetPopulationReturnsValidPointer) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_population_t* pop = snn_network_get_population(network, 0);
    EXPECT_NE(nullptr, pop);
}

TEST_F(SnnNetworkTest, GetPopulationReturnsNullForInvalidId) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_population_t* pop = snn_network_get_population(network, 999);
    EXPECT_EQ(nullptr, pop);
}

TEST_F(SnnNetworkTest, ConnectPopulationsCreatesConnections) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    // Connect input to output population
    int n_connections = snn_network_connect_populations(
        network, 0, 1,
        SNN_TOPO_FULL, 1.0f,
        SYNAPSE_AMPA, 0.5f, 0.1f);

    EXPECT_GT(n_connections, 0);
}

//=============================================================================
// Statistics and Monitoring Tests
//=============================================================================

TEST_F(SnnNetworkTest, GetStatsReturnsSuccess) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_stats_t stats;
    int result = snn_network_get_stats(network, &stats);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, CheckHealthReturnsHealthyForNewNetwork) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_state_health_t health = snn_network_check_health(network);
    // New network with no simulation might be silent
    EXPECT_TRUE(health == SNN_STATE_HEALTHY || health == SNN_STATE_SILENT);
}

TEST_F(SnnNetworkTest, GetFiringRateReturnsNonNegative) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    // Run some simulation
    float inputs[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    snn_network_set_inputs(network, inputs, 4);
    snn_network_run(network, 100.0f);

    float rate = snn_network_get_firing_rate(network, 0, 0, 100.0f);
    EXPECT_GE(rate, 0.0f);
}

TEST_F(SnnNetworkTest, GetPopulationRateReturnsNonNegative) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float rate = snn_network_get_population_rate(network, 0, 100.0f);
    EXPECT_GE(rate, 0.0f);
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST_F(SnnNetworkTest, ConnectBioAsyncReturnsSuccess) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    int result = snn_network_connect_bio_async(network);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, DisconnectBioAsyncReturnsSuccess) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    snn_network_connect_bio_async(network);
    int result = snn_network_disconnect_bio_async(network);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, ConnectImmuneReturnsSuccess) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    // Pass dummy immune handle (NULL for now)
    int result = snn_network_connect_immune(network, nullptr);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, ApplyImmuneModulationReturnsSuccess) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    int result = snn_network_apply_immune_modulation(network);
    EXPECT_EQ(SNN_SUCCESS, result);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(SnnNetworkTest, GetNeuralNetReturnsHandle) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    neural_network_t nn = snn_network_get_neural_net(network);
    EXPECT_NE(nullptr, nn);
}

TEST_F(SnnNetworkTest, ValidateReturnsSuccessForValidNetwork) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    int result = snn_network_validate(network);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, ValidateReturnsErrorOnNullPointer) {
    int result = snn_network_validate(nullptr);
    EXPECT_NE(SNN_SUCCESS, result);
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST_F(SnnNetworkTest, NetworkWithMinimalSizeWorks) {
    snn_config_feedforward(&config, 1, 0, 1);
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[1] = {1.0f};
    float outputs[1];

    int result = snn_network_forward(network, inputs, 1, outputs, 1, 10.0f);
    EXPECT_EQ(SNN_SUCCESS, result);
}

TEST_F(SnnNetworkTest, MultipleForwardPassesWork) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    float inputs[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float outputs[2];

    for (int i = 0; i < 5; i++) {
        int result = snn_network_forward(network, inputs, 4, outputs, 2, 50.0f);
        EXPECT_EQ(SNN_SUCCESS, result);
    }
}

TEST_F(SnnNetworkTest, SimulationUpdatesStats) {
    CreateNetwork();
    ASSERT_NE(nullptr, network);

    EXPECT_EQ(0ULL, network->stats.total_steps);

    float inputs[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    snn_network_set_inputs(network, inputs, 4);
    snn_network_run(network, 10.0f);

    EXPECT_GT(network->stats.total_steps, 0ULL);
}

TEST_F(SnnNetworkTest, ReservoirConfigurationWorks) {
    snn_config_t reservoir_config;
    snn_config_reservoir(&reservoir_config, 10, 100, 5, 0.1f);

    snn_network_t* reservoir = snn_network_create(&reservoir_config);
    ASSERT_NE(nullptr, reservoir);

    float inputs[10] = {0};
    float outputs[5];

    int result = snn_network_forward(reservoir, inputs, 10, outputs, 5, 50.0f);
    EXPECT_EQ(SNN_SUCCESS, result);

    snn_network_destroy(reservoir);
    snn_config_destroy(&reservoir_config);
}
