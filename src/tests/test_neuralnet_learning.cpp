#include "test_helpers.h"

// Test STDP learning
TEST(NeuralNetLearning, STDP) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    
    ASSERT_NE(network, nullptr);
    
    // Add test connection
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));
    
    // Record spikes with appropriate timing for STDP
    neural_network_record_spike(network, 0, 1.0f, 1);
    neural_network_record_spike(network, 1, 1.0f, 2);
    
    // Apply STDP
    uint32_t modified = neural_network_apply_stdp(network, 0, 2);
    
    // Verify weight changes
    ASSERT_GT(modified, 0);
    
    neural_network_destroy(network);
}

// Test Oja's learning rule
TEST(NeuralNetLearning, Oja) {
    network_config_t config = create_test_config();
    neural_network_t network = neural_network_create(&config);
    
    ASSERT_NE(network, nullptr);
    
    // Add test connection
    ASSERT_TRUE(neural_network_add_connection(network, 0, 1, 0.5f));
    
    // Activate neurons
    neural_network_update_neuron(network, 0, 1.0f, 1);
    neural_network_update_neuron(network, 1, 1.0f, 1);
    
    // Apply Oja's rule
    uint32_t modified = neural_network_apply_oja(network, 0, 1);
    
    // Verify weight changes
    ASSERT_GT(modified, 0);
    
    neural_network_destroy(network);
}
