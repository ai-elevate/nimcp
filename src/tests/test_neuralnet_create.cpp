#include "test_helpers.h"

//=============================================================================
// Network Creation Tests
//=============================================================================

// Test network creation with valid configuration
TEST(NeuralNetCreate, ValidConfig) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);

    ASSERT_NE(network, nullptr);

    // Verify initial network state
    float state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, 0, &state));
    ASSERT_TRUE(float_equals(state, -65.0f)); // Default rest potential

    neural_network_destroy(network);
}

// Test network creation with nullptr configuration
TEST(NeuralNetCreate, NullConfig) {
    neural_network_t network = neural_network_create(nullptr);
    ASSERT_EQ(network, nullptr);
}

// Test network creation with invalid neuron count
TEST(NeuralNetCreate, InvalidNeuronCount) {
    network_config_t config = create_test_config();
    config.num_neurons = MAX_NEURONS + 1;

    neural_network_t network = neural_network_create(&config);
    ASSERT_EQ(network, nullptr);
}

// Test network creation with zero neurons
TEST(NeuralNetCreate, ZeroNeurons) {
    network_config_t config = create_test_config();
    config.num_neurons = 0;

    neural_network_t network = neural_network_create(&config);
    ASSERT_EQ(network, nullptr);
}

// Test proper initialization of neuron parameters
TEST(NeuralNetCreate, NeuronInitialization) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);

    ASSERT_NE(network, nullptr);

    // Check first neuron's parameters
    float state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, 0, &state));
    ASSERT_TRUE(float_equals(state, -65.0f));

    // Check neuron count
    network_stats_t stats;
    neural_network_get_stats(network, &stats);
    ASSERT_EQ(stats.num_neurons, config.num_neurons);

    // Verify E/I ratio
    uint32_t expected_inhibitory = (uint32_t)(config.num_neurons * (1.0f - config.ei_ratio));
    ASSERT_EQ(stats.num_inhibitory, expected_inhibitory);

    neural_network_destroy(network);
}

//=============================================================================
// Low-Level Neuron Function Tests
//=============================================================================

// Test neuron activation computation with different activation types
TEST(NeuralNetNeuron, ActivationFunctions) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add a neuron with sigmoid activation
    uint32_t sigmoid_id = neural_network_add_neuron(network, ACTIVATION_SIGMOID);
    ASSERT_NE(sigmoid_id, UINT32_MAX);

    // Add a neuron with ReLU activation
    uint32_t relu_id = neural_network_add_neuron(network, ACTIVATION_RELU);
    ASSERT_NE(relu_id, UINT32_MAX);

    // Add a neuron with tanh activation
    uint32_t tanh_id = neural_network_add_neuron(network, ACTIVATION_TANH);
    ASSERT_NE(tanh_id, UINT32_MAX);

    // Update neurons with positive input
    ASSERT_TRUE(neural_network_update_neuron(network, sigmoid_id, 2.0f, 1));
    ASSERT_TRUE(neural_network_update_neuron(network, relu_id, 2.0f, 1));
    ASSERT_TRUE(neural_network_update_neuron(network, tanh_id, 2.0f, 1));

    // Verify different activation results
    float sigmoid_state, relu_state, tanh_state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, sigmoid_id, &sigmoid_state));
    ASSERT_TRUE(neural_network_get_neuron_state(network, relu_id, &relu_state));
    ASSERT_TRUE(neural_network_get_neuron_state(network, tanh_id, &tanh_state));

    // Sigmoid should be in (0, 1)
    EXPECT_GT(sigmoid_state, 0.0f);
    EXPECT_LT(sigmoid_state, 1.0f);

    // ReLU should be positive
    EXPECT_GT(relu_state, 0.0f);

    // Tanh should be in (-1, 1)
    EXPECT_GT(tanh_state, -1.0f);
    EXPECT_LT(tanh_state, 1.0f);

    neural_network_destroy(network);
}

// Test neuron state update and retrieval
TEST(NeuralNetNeuron, StateUpdate) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Update neuron state
    ASSERT_TRUE(neural_network_update_neuron(network, 0, 1.0f, 1));

    // Verify state was updated
    float state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, 0, &state));
    EXPECT_NE(state, -65.0f); // Should not be rest potential anymore

    // Test invalid neuron ID
    ASSERT_FALSE(neural_network_get_neuron_state(network, MAX_NEURONS + 1, &state));

    neural_network_destroy(network);
}

// Test neuron refractory period
TEST(NeuralNetNeuron, RefractoryPeriod) {
    network_config_t config = create_test_config();
    config.refractory_period = 5.0f;
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Fire neuron
    ASSERT_TRUE(neural_network_record_spike(network, 0, 1.0f, 1));

    // Try to update during refractory period
    ASSERT_FALSE(neural_network_update_neuron(network, 0, 1.0f, 2));

    // Update after refractory period
    ASSERT_TRUE(neural_network_update_neuron(network, 0, 1.0f, 10));

    neural_network_destroy(network);
}

// Test synaptic connection creation
TEST(NeuralNetNeuron, SynapticConnections) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connection between neurons
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));

    // Test adding another connection
    ASSERT_TRUE(neural_network_add_connection(network, 0, 2, 0.7f));

    // Test weight clamping (should clamp to max_weight)
    ASSERT_TRUE(neural_network_add_connection(network, 1, 2, 100.0f));

    // Test invalid connection (same neuron)
    // Note: This depends on implementation - might be allowed

    // Test connection to non-existent neuron
    ASSERT_FALSE(neural_network_add_connection(network, 0, MAX_NEURONS + 1, 0.5f));

    neural_network_destroy(network);
}

// Test weight normalization
TEST(NeuralNetNeuron, WeightNormalization) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add several connections
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 1.0f));
    ASSERT_TRUE(neural_network_add_connection(network, 0, 2, 1.0f));
    ASSERT_TRUE(neural_network_add_connection(network, 0, 3, 1.0f));

    // Get weight norm before normalization
    float norm_before = neural_network_get_weight_norm(network, 0);
    EXPECT_GT(norm_before, 0.0f);

    // Normalize weights
    ASSERT_TRUE(neural_network_normalize_weights(network, 0));

    // Get weight norm after normalization
    float norm_after = neural_network_get_weight_norm(network, 0);

    // Norm should be closer to target (1.0 by default)
    EXPECT_LT(fabsf(norm_after - 1.0f), fabsf(norm_before - 1.0f));

    neural_network_destroy(network);
}

// Test weight statistics computation
TEST(NeuralNetNeuron, WeightStatistics) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connections with known weights
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));
    ASSERT_TRUE(neural_network_add_connection(network, 0, 2, 0.3f));
    ASSERT_TRUE(neural_network_add_connection(network, 0, 3, 0.7f));

    // Get weight statistics
    float mean, std_dev;
    neural_network_get_weight_statistics(network, 0, &mean, &std_dev);

    // Mean should be (0.5 + 0.3 + 0.7) / 3 = 0.5
    EXPECT_TRUE(float_equals(mean, 0.5f));

    // Std dev should be positive for non-uniform weights
    EXPECT_GT(std_dev, 0.0f);

    neural_network_destroy(network);
}

// Test average activity computation
TEST(NeuralNetNeuron, AverageActivity) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Initial activity should be zero
    float activity = neural_network_get_average_activity(network, 0);
    EXPECT_TRUE(float_equals(activity, 0.0f));

    // Record some spikes
    for (uint64_t t = 1; t <= 10; t++) {
        neural_network_record_spike(network, 0, 1.0f, t);
    }

    // Activity should now be positive
    activity = neural_network_get_average_activity(network, 0);
    EXPECT_GT(activity, 0.0f);

    neural_network_destroy(network);
}

// Test threshold adaptation
TEST(NeuralNetNeuron, ThresholdAdaptation) {
    network_config_t config = create_test_config();
    config.adaptation_rate = 0.1f;
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Adapt threshold based on activity
    ASSERT_TRUE(neural_network_adapt_threshold(network, 0, 1));

    // The threshold should be adjusted (exact value depends on implementation)
    // Just verify the function executes without error

    neural_network_destroy(network);
}

// Test synaptic trace updates
TEST(NeuralNetNeuron, SynapticTraces) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connection
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));

    // Update traces
    neural_network_update_traces(network, 0, 1);

    // Fire neuron to activate trace
    neural_network_record_spike(network, 0, 1.0f, 2);

    // Update traces again
    neural_network_update_traces(network, 0, 3);

    // Verify function executes without error

    neural_network_destroy(network);
}

// Test dynamic neuron addition
TEST(NeuralNetNeuron, DynamicNeuronAddition) {
    network_config_t config = create_test_config();
    config.num_neurons = 5; // Start with small network
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Get initial stats
    network_stats_t stats_before;
    neural_network_get_stats(network, &stats_before);
    EXPECT_EQ(stats_before.num_neurons, 5);

    // Add new neuron
    uint32_t new_id = neural_network_add_neuron(network, ACTIVATION_SIGMOID);
    ASSERT_NE(new_id, UINT32_MAX);

    // Get updated stats
    network_stats_t stats_after;
    neural_network_get_stats(network, &stats_after);
    EXPECT_EQ(stats_after.num_neurons, 6);

    // Verify new neuron works
    ASSERT_TRUE(neural_network_update_neuron(network, new_id, 1.0f, 1));

    float state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, new_id, &state));

    neural_network_destroy(network);
}

// Test network reset functionality
TEST(NeuralNetNeuron, NetworkReset) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Activate neurons and record spikes
    neural_network_update_neuron(network, 0, 1.0f, 1);
    neural_network_record_spike(network, 0, 1.0f, 1);
    neural_network_update_neuron(network, 1, 1.0f, 1);

    // Reset network
    neural_network_reset(network);

    // Verify neurons are back to rest potential
    float state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, 0, &state));
    EXPECT_TRUE(float_equals(state, -65.0f));

    // Verify activity is reset
    float activity = neural_network_get_average_activity(network, 0);
    EXPECT_TRUE(float_equals(activity, 0.0f));

    neural_network_destroy(network);
}

// Test compute step functionality
TEST(NeuralNetNeuron, ComputeStep) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connections
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));
    ASSERT_TRUE(neural_network_add_connection(network, 1, 2, 0.5f));

    // Activate first neuron
    neural_network_update_neuron(network, 0, 5.0f, 1);

    // Run compute step
    uint32_t active = neural_network_compute_step(network, 2);

    // Some neurons should be active
    EXPECT_GT(active, 0);

    neural_network_destroy(network);
}

// Test forward pass functionality (NIMCP 2.5)
TEST(NeuralNetNeuron, ForwardPass) {
    network_config_t config = create_test_config();
    config.num_neurons = 10;
    config.input_size = 3;
    config.output_size = 2;

    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Create input
    float input[3] = {0.5f, 0.7f, 0.3f};
    float output[2] = {0.0f, 0.0f};

    // Run forward pass
    ASSERT_TRUE(neural_network_forward(network, input, 3, output, 2));

    // Output should be modified
    EXPECT_TRUE(output[0] != 0.0f || output[1] != 0.0f);

    neural_network_destroy(network);
}

//=============================================================================
// Low-Level Neuron Function Tests (Additional Coverage)
//=============================================================================

// Test membrane potential computation with synaptic inputs
TEST(NeuralNetLowLevel, MembranePotential) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connections to create synaptic inputs
    ASSERT_TRUE(neural_network_add_connection(network, 1, 0, 0.8f));
    ASSERT_TRUE(neural_network_add_connection(network, 2, 0, 0.6f));

    // Activate presynaptic neurons
    neural_network_update_neuron(network, 1, 3.0f, 1);
    neural_network_update_neuron(network, 2, 2.0f, 1);

    // Update target neuron - should integrate synaptic inputs
    neural_network_update_neuron(network, 0, 0.0f, 2);

    // Verify neuron state was affected by synaptic inputs
    float state;
    ASSERT_TRUE(neural_network_get_neuron_state(network, 0, &state));
    EXPECT_NE(state, -65.0f); // Should not be at rest potential

    neural_network_destroy(network);
}

// Test calcium dynamics
TEST(NeuralNetLowLevel, CalciumDynamics) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Record multiple spikes to increase calcium
    for (int i = 0; i < 5; i++) {
        neural_network_record_spike(network, 0, 1.0f, i + 1);
    }

    // Run compute step to trigger calcium updates
    neural_network_compute_step(network, 10);

    // Calcium should have accumulated and then decayed
    // (exact verification requires exposing calcium value)

    neural_network_destroy(network);
}

// Test spike propagation through synapses
TEST(NeuralNetLowLevel, SpikePropagation) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Create a chain: 0 -> 1 -> 2
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.9f));
    ASSERT_TRUE(neural_network_add_connection(network, 1, 2, 0.9f));

    // Fire first neuron
    neural_network_record_spike(network, 0, 2.0f, 1);

    // Run compute steps to allow propagation
    neural_network_compute_step(network, 2);
    neural_network_compute_step(network, 3);

    // Check that downstream neurons were affected
    float state1, state2;
    ASSERT_TRUE(neural_network_get_neuron_state(network, 1, &state1));
    ASSERT_TRUE(neural_network_get_neuron_state(network, 2, &state2));

    // At least one should be affected (exact behavior depends on activation)
    EXPECT_TRUE(state1 != -65.0f || state2 != -65.0f);

    neural_network_destroy(network);
}

// Test excitatory vs inhibitory neuron behavior
TEST(NeuralNetLowLevel, ExcitatoryInhibitory) {
    network_config_t config = create_test_config();
    config.num_neurons = 10;
    config.ei_ratio = 0.8f; // 80% excitatory, 20% inhibitory
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Get stats to verify E/I ratio
    network_stats_t stats;
    neural_network_get_stats(network, &stats);

    uint32_t expected_inhibitory = (uint32_t)(config.num_neurons * (1.0f - config.ei_ratio));
    uint32_t expected_excitatory = config.num_neurons - expected_inhibitory;

    EXPECT_EQ(stats.num_inhibitory, expected_inhibitory);
    EXPECT_EQ(stats.num_excitatory, expected_excitatory);

    neural_network_destroy(network);
}

// Test synaptic strength modulation
TEST(NeuralNetLowLevel, SynapticStrength) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connection
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));

    // Record spike to activate trace and strength updates
    neural_network_record_spike(network, 0, 1.0f, 1);

    // Update traces
    neural_network_update_traces(network, 0, 5);

    // Synaptic strength should be maintained
    // (verification requires exposing strength value)

    neural_network_destroy(network);
}

// Test meta-plasticity computation
TEST(NeuralNetLowLevel, MetaPlasticity) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connection
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));

    // Create activity pattern with variance
    for (uint64_t t = 1; t <= 50; t++) {
        float input = (t % 2 == 0) ? 1.0f : 0.1f; // Alternating activity
        neural_network_update_neuron(network, 0, input, t);
        neural_network_update_plasticity(network, 0, t);
    }

    // Meta-plasticity should have been updated based on activity variance
    // (exact verification requires exposing meta-plasticity values)

    neural_network_destroy(network);
}

// Test spike history recording
TEST(NeuralNetLowLevel, SpikeHistory) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Record multiple spikes
    for (uint64_t t = 1; t <= 10; t++) {
        neural_network_record_spike(network, 0, 1.0f + t * 0.1f, t);
    }

    // Average activity should reflect spike history
    float activity = neural_network_get_average_activity(network, 0);
    EXPECT_GT(activity, 0.0f);

    neural_network_destroy(network);
}

// Test activation function strategy dispatch
TEST(NeuralNetLowLevel, ActivationStrategyDispatch) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Test all activation types
    activation_type_t types[] = {
        ACTIVATION_SIGMOID,
        ACTIVATION_TANH,
        ACTIVATION_RELU,
        ACTIVATION_LEAKY_RELU,
        ACTIVATION_ADAPTIVE
    };

    for (int i = 0; i < 5; i++) {
        uint32_t neuron_id = neural_network_add_neuron(network, types[i]);
        ASSERT_NE(neuron_id, UINT32_MAX);

        // Update with same input
        neural_network_update_neuron(network, neuron_id, 1.5f, 1);

        // Get state - should be different for different activation functions
        float state;
        ASSERT_TRUE(neural_network_get_neuron_state(network, neuron_id, &state));
    }

    neural_network_destroy(network);
}

// Test network maintenance routine
TEST(NeuralNetLowLevel, MaintenanceRoutine) {
    network_config_t config = create_test_config();
    config.update_interval = 50;
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connections
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.05f)); // Weak
    ASSERT_TRUE(neural_network_add_connection(network, 0, 2, 0.8f));  // Strong

    // Run some activity
    for (uint64_t t = 1; t <= 30; t++) {
        neural_network_update_neuron(network, 0, 1.0f, t);
    }

    // Run maintenance
    neural_network_maintain(network, 100);

    // Network should still be functional
    network_stats_t stats;
    neural_network_get_stats(network, &stats);
    EXPECT_GT(stats.num_neurons, 0);

    neural_network_destroy(network);
}

// Test homeostasis maintenance
TEST(NeuralNetLowLevel, HomeostasisMaintenance) {
    network_config_t config = create_test_config();
    config.homeostatic_rate = 0.1f;
    config.target_activity = 0.5f;
    config.update_interval = 50;
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Create sustained activity
    for (uint64_t t = 1; t <= 60; t++) {
        neural_network_update_neuron(network, 0, 2.0f, t);
        if (t % 10 == 0) {
            neural_network_record_spike(network, 0, 1.0f, t);
        }
    }

    // Run homeostasis maintenance
    neural_network_maintain_homeostasis(network, 100);

    // Network should maintain stats
    network_stats_t stats;
    neural_network_get_stats(network, &stats);
    EXPECT_GT(stats.network_stability, 0.0f);

    neural_network_destroy(network);
}

// Test synaptic pruning with threshold
TEST(NeuralNetLowLevel, SynapticPruningThreshold) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add connections with varying weights
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.001f)); // Very weak
    ASSERT_TRUE(neural_network_add_connection(network, 0, 2, 0.005f)); // Weak
    ASSERT_TRUE(neural_network_add_connection(network, 0, 3, 0.5f));   // Strong

    // Prune with threshold that should remove first two
    uint32_t pruned = neural_network_prune_synapses(network, 0.01f);
    EXPECT_GE(pruned, 1); // At least one should be pruned

    neural_network_destroy(network);
}

// Test neuron dump function (debug utility)
TEST(NeuralNetLowLevel, NeuronDump) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    ASSERT_NE(network, nullptr);

    // Add some connections
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));
    ASSERT_TRUE(neural_network_add_connection(network, 0, 2, 0.7f));

    // Activate neuron
    neural_network_update_neuron(network, 0, 1.0f, 1);

    // Dump neuron (should not crash, output to stdout)
    neural_network_dump_neuron(network, 0);

    // Test with invalid ID (should handle gracefully)
    neural_network_dump_neuron(network, MAX_NEURONS + 1);

    neural_network_destroy(network);
}
