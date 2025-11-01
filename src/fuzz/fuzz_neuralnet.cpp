/**
 * @file fuzz_neuralnet.cpp
 * @brief Fuzzing target for neural network API
 *
 * Tests neural network creation, configuration, and basic operations
 * with randomly generated inputs to discover crashes, memory errors,
 * and undefined behavior.
 *
 * Build:
 *   cmake -DENABLE_FUZZING=ON ..
 *   make fuzz_neuralnet
 *
 * Run:
 *   ./fuzz_neuralnet -max_total_time=300
 *   ./fuzz_neuralnet corpus/ -max_total_time=3600
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "nimcp_neuralnet.h"

// Minimum input size needed
#define MIN_INPUT_SIZE sizeof(nimcp_neuralnet_config_t)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    // Need at least enough bytes for a config structure
    if (size < MIN_INPUT_SIZE) {
        return 0;
    }

    // Copy fuzzer data into config structure
    nimcp_neuralnet_config_t config;
    memcpy(&config, data, sizeof(config));

    // Sanitize config values to prevent extreme allocations
    // while still allowing fuzzer to explore edge cases
    config.num_inputs = config.num_inputs % 1000;
    config.num_outputs = config.num_outputs % 1000;
    config.num_hidden = config.num_hidden % 1000;

    // Clamp learning rate to reasonable range
    if (config.learning_rate < 0.0f || config.learning_rate > 1.0f) {
        config.learning_rate = 0.01f;
    }

    // Clamp other float parameters
    if (config.stdp_a_plus < 0.0f || config.stdp_a_plus > 1.0f) {
        config.stdp_a_plus = 0.01f;
    }
    if (config.stdp_a_minus < 0.0f || config.stdp_a_minus > 1.0f) {
        config.stdp_a_minus = 0.01f;
    }

    // Try to create network
    nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);

    if (net != nullptr) {
        // If creation succeeded, try some operations

        // Try forward pass with remaining fuzzer data
        size_t remaining = size - MIN_INPUT_SIZE;
        if (remaining >= config.num_inputs * sizeof(float)) {
            const float* inputs = reinterpret_cast<const float*>(data + MIN_INPUT_SIZE);
            float outputs[1000];  // Max outputs clamped above

            nimcp_neuralnet_forward(net, inputs, outputs);
        }

        // Try getting stats
        network_stats_t stats;
        nimcp_neuralnet_get_stats(net, &stats);

        // Clean up
        nimcp_neuralnet_destroy(net);
    }

    // Test null pointer handling
    nimcp_neuralnet_destroy(nullptr);

    return 0;
}
