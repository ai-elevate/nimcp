#include "test_helpers.h"

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
