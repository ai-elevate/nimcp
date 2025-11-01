/**
 * @file test_stress.cpp
 * @brief Stress and load testing for NIMCP components
 *
 * Tests system behavior under extreme conditions:
 * - High concurrent load
 * - Memory pressure
 * - Long-running operations
 * - Resource exhaustion scenarios
 *
 * These tests ensure the system remains stable and secure under stress.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "../include/nimcp_brain.h"
#include "../include/nimcp_neuralnet.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_queue_manager.h"

//==============================================================================
// Test Configuration
//==============================================================================

constexpr int STRESS_ITERATIONS = 1000;
constexpr int CONCURRENT_THREADS = 8;
constexpr int STRESS_DURATION_SECONDS = 5;

//==============================================================================
// Test Fixture
//==============================================================================

class StressTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Initialize test environment
    }

    void TearDown() override
    {
        // Cleanup
    }
};

//==============================================================================
// Neural Network Stress Tests
//==============================================================================

/**
 * @test High-volume neural network creation and destruction
 */
TEST_F(StressTest, NeuralNetwork_MassCreationDestruction)
{
    nimcp_neuralnet_config_t config = {.num_inputs = 10,
                                       .num_outputs = 5,
                                       .num_hidden = 20,
                                       .learning_rate = 0.01f,
                                       .stdp_a_plus = 0.01f,
                                       .stdp_a_minus = 0.01f,
                                       .tau_plus = 20.0f,
                                       .tau_minus = 20.0f};

    std::vector<nimcp_neuralnet_t*> networks;
    networks.reserve(STRESS_ITERATIONS);

    // Rapid creation
    for (int i = 0; i < STRESS_ITERATIONS; i++) {
        nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
        ASSERT_NE(net, nullptr) << "Failed to create network " << i;
        networks.push_back(net);
    }

    // Rapid destruction
    for (auto* net : networks) {
        nimcp_neuralnet_destroy(net);
    }
}

/**
 * @test Concurrent neural network operations
 */
TEST_F(StressTest, NeuralNetwork_ConcurrentOperations)
{
    nimcp_neuralnet_config_t config = {.num_inputs = 10,
                                       .num_outputs = 5,
                                       .num_hidden = 20,
                                       .learning_rate = 0.01f,
                                       .stdp_a_plus = 0.01f,
                                       .stdp_a_minus = 0.01f,
                                       .tau_plus = 20.0f,
                                       .tau_minus = 20.0f};

    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;

    // Launch concurrent workers
    for (int t = 0; t < CONCURRENT_THREADS; t++) {
        threads.emplace_back([&config, &error_count]() {
            // Each thread creates network and performs operations
            nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
            if (!net) {
                error_count++;
                return;
            }

            float inputs[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
            float outputs[5] = {0};

            // Perform many forward passes
            for (int i = 0; i < 100; i++) {
                nimcp_neuralnet_forward(net, inputs, outputs);

                // Verify outputs are in valid range
                for (int j = 0; j < 5; j++) {
                    if (std::isnan(outputs[j]) || std::isinf(outputs[j])) {
                        error_count++;
                    }
                }
            }

            nimcp_neuralnet_destroy(net);
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(error_count.load(), 0) << "Errors occurred during concurrent operations";
}

/**
 * @test Long-running neural network stability
 */
TEST_F(StressTest, DISABLED_NeuralNetwork_LongRunningStability)
{
    nimcp_neuralnet_config_t config = {.num_inputs = 10,
                                       .num_outputs = 5,
                                       .num_hidden = 20,
                                       .learning_rate = 0.01f,
                                       .stdp_a_plus = 0.01f,
                                       .stdp_a_minus = 0.01f,
                                       .tau_plus = 20.0f,
                                       .tau_minus = 20.0f};

    nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
    ASSERT_NE(net, nullptr);

    float inputs[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float outputs[5] = {0};

    auto start = std::chrono::steady_clock::now();
    auto end = start + std::chrono::seconds(STRESS_DURATION_SECONDS);

    int iteration_count = 0;
    while (std::chrono::steady_clock::now() < end) {
        nimcp_neuralnet_forward(net, inputs, outputs);

        // Verify outputs remain stable
        for (int j = 0; j < 5; j++) {
            ASSERT_FALSE(std::isnan(outputs[j])) << "NaN detected at iteration " << iteration_count;
            ASSERT_FALSE(std::isinf(outputs[j])) << "Inf detected at iteration " << iteration_count;
        }

        iteration_count++;
    }

    nimcp_neuralnet_destroy(net);

    // Should complete thousands of iterations
    EXPECT_GT(iteration_count, 1000) << "Unexpectedly low iteration count";
}

//==============================================================================
// Queue Manager Stress Tests
//==============================================================================

/**
 * @test High-throughput queue operations
 */
TEST_F(StressTest, QueueManager_HighThroughput)
{
    nimcp_queue_manager_config_t config = {.num_channels = 4, .max_queue_size = 1000, .num_priorities = 3};

    nimcp_queue_manager_t* manager = nimcp_queue_manager_create(&config);
    ASSERT_NE(manager, nullptr);

    std::atomic<int> enqueue_count{0};
    std::atomic<int> dequeue_count{0};

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    // Producer threads
    for (int t = 0; t < CONCURRENT_THREADS / 2; t++) {
        producers.emplace_back([manager, &enqueue_count]() {
            nimcp_message_t msg = {.size = 64, .data = nullptr};

            uint8_t data[64] = {0};
            msg.data = data;

            for (int i = 0; i < STRESS_ITERATIONS; i++) {
                uint32_t channel = i % 4;
                if (nimcp_queue_manager_enqueue(manager, channel, &msg, 1)) {
                    enqueue_count++;
                }
                std::this_thread::yield();
            }
        });
    }

    // Consumer threads
    for (int t = 0; t < CONCURRENT_THREADS / 2; t++) {
        consumers.emplace_back([manager, &dequeue_count]() {
            nimcp_message_t msg;

            for (int i = 0; i < STRESS_ITERATIONS; i++) {
                uint32_t channel = i % 4;
                if (nimcp_queue_manager_dequeue(manager, channel, &msg, 1)) {
                    dequeue_count++;
                }
                std::this_thread::yield();
            }
        });
    }

    // Wait for completion
    for (auto& t : producers)
        t.join();
    for (auto& t : consumers)
        t.join();

    nimcp_queue_manager_destroy(manager);

    // Verify high throughput
    EXPECT_GT(enqueue_count.load(), STRESS_ITERATIONS) << "Low enqueue throughput";
    EXPECT_GT(dequeue_count.load(), 0) << "No messages dequeued";
}

/**
 * @test Queue manager under memory pressure
 */
TEST_F(StressTest, QueueManager_MemoryPressure)
{
    // Create multiple queue managers to simulate memory pressure
    std::vector<nimcp_queue_manager_t*> managers;

    nimcp_queue_manager_config_t config = {.num_channels = 8, .max_queue_size = 500, .num_priorities = 3};

    // Create as many as we can
    for (int i = 0; i < 20; i++) {
        nimcp_queue_manager_t* manager = nimcp_queue_manager_create(&config);
        if (manager != nullptr) {
            managers.push_back(manager);
        } else {
            // Allocation failure is acceptable under memory pressure
            break;
        }
    }

    EXPECT_GT(managers.size(), 0) << "Could not create any queue managers";

    // Test that all managers still work
    for (auto* manager : managers) {
        nimcp_message_t msg = {.size = 64, .data = nullptr};
        uint8_t data[64] = {0};
        msg.data = data;

        EXPECT_NO_FATAL_FAILURE(nimcp_queue_manager_enqueue(manager, 0, &msg, 1));
    }

    // Cleanup
    for (auto* manager : managers) {
        nimcp_queue_manager_destroy(manager);
    }
}

//==============================================================================
// Memory Allocation Stress Tests
//==============================================================================

/**
 * @test Rapid allocation and deallocation
 */
TEST_F(StressTest, Memory_RapidAllocationDeallocation)
{
    std::vector<void*> allocations;
    allocations.reserve(STRESS_ITERATIONS);

    // Allocate
    for (int i = 0; i < STRESS_ITERATIONS; i++) {
        size_t size = 64 + (i % 1024);  // Variable sizes
        void* ptr = malloc(size);
        ASSERT_NE(ptr, nullptr) << "Allocation failed at iteration " << i;
        allocations.push_back(ptr);
    }

    // Deallocate
    for (void* ptr : allocations) {
        free(ptr);
    }
}

/**
 * @test Fragmentation resistance
 */
TEST_F(StressTest, Memory_FragmentationResistance)
{
    std::vector<void*> allocations;
    const int ALLOC_COUNT = 500;

    // Allocate mix of sizes
    for (int i = 0; i < ALLOC_COUNT; i++) {
        size_t size;
        if (i % 3 == 0)
            size = 16;
        else if (i % 3 == 1)
            size = 256;
        else
            size = 1024;

        void* ptr = malloc(size);
        if (ptr != nullptr) {
            allocations.push_back(ptr);
        }
    }

    // Free every other allocation
    for (size_t i = 0; i < allocations.size(); i += 2) {
        free(allocations[i]);
        allocations[i] = nullptr;
    }

    // Try to allocate again (tests fragmentation handling)
    for (size_t i = 0; i < allocations.size(); i += 2) {
        allocations[i] = malloc(128);
        // OK if this fails under fragmentation
    }

    // Cleanup
    for (void* ptr : allocations) {
        if (ptr != nullptr) {
            free(ptr);
        }
    }
}

//==============================================================================
// Concurrent Brain Operations
//==============================================================================

/**
 * @test Concurrent brain creation and destruction
 */
TEST_F(StressTest, Brain_ConcurrentLifecycle)
{
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < CONCURRENT_THREADS; t++) {
        threads.emplace_back([&success_count, &error_count]() {
            nimcp_brain_config_t config = {
                .num_inputs = 10,
                .num_outputs = 5,
                .hidden_layers = {20, 15},
                .num_hidden_layers = 2,
                .consolidation_interval = 100,
                .salience_threshold = 0.5f,
            };
            // Generate unique task name
            static std::atomic<int> thread_counter{0};
            snprintf(config.task_name, sizeof(config.task_name), "stress_test_%d",
                     thread_counter.fetch_add(1));

            nimcp_brain_t* brain = nimcp_brain_create(&config);
            if (brain == nullptr) {
                error_count++;
                return;
            }

            // Perform some operations
            float inputs[10] = {0};
            float outputs[5] = {0};

            for (int i = 0; i < 10; i++) {
                nimcp_brain_process(brain, inputs, outputs);
            }

            nimcp_brain_destroy(brain);
            success_count++;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_GT(success_count.load(), 0) << "No successful brain operations";
    EXPECT_EQ(error_count.load(), 0) << "Errors during concurrent brain operations";
}

//==============================================================================
// Performance Regression Tests
//==============================================================================

/**
 * @test Ensure operations complete within reasonable time
 */
TEST_F(StressTest, Performance_OperationLatency)
{
    nimcp_neuralnet_config_t config = {.num_inputs = 10,
                                       .num_outputs = 5,
                                       .num_hidden = 20,
                                       .learning_rate = 0.01f,
                                       .stdp_a_plus = 0.01f,
                                       .stdp_a_minus = 0.01f,
                                       .tau_plus = 20.0f,
                                       .tau_minus = 20.0f};

    nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
    ASSERT_NE(net, nullptr);

    float inputs[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float outputs[5] = {0};

    // Measure 1000 forward passes
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        nimcp_neuralnet_forward(net, inputs, outputs);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    nimcp_neuralnet_destroy(net);

    // Should complete in reasonable time (< 100ms for 1000 iterations)
    EXPECT_LT(duration.count(), 100000) << "Operations too slow: " << duration.count() << " us";

    double avg_latency_us = duration.count() / 1000.0;
    std::cout << "Average forward pass latency: " << avg_latency_us << " µs" << std::endl;
}
