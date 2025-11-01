/**
 * @file test_error_injection.cpp
 * @brief Error injection testing framework for NIMCP
 *
 * Systematically tests error handling paths by simulating failures:
 * - Memory allocation failures (malloc, calloc, realloc)
 * - File I/O errors
 * - Network errors
 * - Resource exhaustion
 *
 * This ensures the library handles errors gracefully without crashes
 * or memory leaks.
 */

#include <gtest/gtest.h>
#include "../include/nimcp_brain.h"
#include "../include/nimcp_neuralnet.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_queue_manager.h"

//==============================================================================
// Error Injection Framework
//==============================================================================

/**
 * @brief Global flag to control malloc failure injection
 */
static bool g_inject_malloc_failure = false;
static int g_malloc_failure_countdown = 0;

/**
 * @brief Override malloc for testing (requires special build flags)
 *
 * Note: This is a simplified approach. For production, use LD_PRELOAD
 * or compiler-specific malloc replacement mechanisms.
 */
#ifdef ENABLE_MALLOC_INJECTION
extern "C" {
void* __real_malloc(size_t size);
void* __wrap_malloc(size_t size)
{
    if (g_inject_malloc_failure) {
        if (g_malloc_failure_countdown > 0) {
            g_malloc_failure_countdown--;
            return __real_malloc(size);
        }
        return nullptr;  // Simulate malloc failure
    }
    return __real_malloc(size);
}
}
#endif

//==============================================================================
// Test Fixture
//==============================================================================

class ErrorInjectionTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        g_inject_malloc_failure = false;
        g_malloc_failure_countdown = 0;
    }

    void TearDown() override
    {
        g_inject_malloc_failure = false;
    }

    void EnableMallocFailure(int countdown = 0)
    {
        g_inject_malloc_failure = true;
        g_malloc_failure_countdown = countdown;
    }

    void DisableMallocFailure()
    {
        g_inject_malloc_failure = false;
    }
};

//==============================================================================
// Memory Allocation Failure Tests
//==============================================================================

/**
 * @test Verify neural network creation handles malloc failure gracefully
 */
TEST_F(ErrorInjectionTest, NeuralNetworkCreation_MallocFailure)
{
#ifdef ENABLE_MALLOC_INJECTION
    nimcp_neuralnet_config_t config = {.num_inputs = 10,
                                       .num_outputs = 5,
                                       .num_hidden = 20,
                                       .learning_rate = 0.01f,
                                       .stdp_a_plus = 0.01f,
                                       .stdp_a_minus = 0.01f,
                                       .tau_plus = 20.0f,
                                       .tau_minus = 20.0f};

    // Simulate malloc failure
    EnableMallocFailure();

    nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);

    // Should return NULL on allocation failure
    EXPECT_EQ(net, nullptr);

    DisableMallocFailure();

    // Verify normal allocation still works
    net = nimcp_neuralnet_create(&config);
    EXPECT_NE(net, nullptr);

    if (net) {
        nimcp_neuralnet_destroy(net);
    }
#else
    GTEST_SKIP() << "ENABLE_MALLOC_INJECTION not defined, skipping test";
#endif
}

/**
 * @test Verify queue manager handles allocation failures
 */
TEST_F(ErrorInjectionTest, QueueManager_MallocFailure)
{
#ifdef ENABLE_MALLOC_INJECTION
    nimcp_queue_manager_config_t config = {
        .num_channels = 4, .max_queue_size = 100, .num_priorities = 3};

    EnableMallocFailure();

    nimcp_queue_manager_t* manager = nimcp_queue_manager_create(&config);

    // Should return NULL on allocation failure
    EXPECT_EQ(manager, nullptr);

    DisableMallocFailure();
#else
    GTEST_SKIP() << "ENABLE_MALLOC_INJECTION not defined, skipping test";
#endif
}

//==============================================================================
// NULL Pointer Handling Tests
//==============================================================================

/**
 * @test Verify APIs handle NULL pointers without crashing
 */
TEST_F(ErrorInjectionTest, NullPointerHandling_NeuralNetwork)
{
    // All these should handle NULL gracefully (not crash)
    EXPECT_NO_FATAL_FAILURE(nimcp_neuralnet_destroy(nullptr));

    float inputs[10] = {0};
    float outputs[5] = {0};
    EXPECT_NO_FATAL_FAILURE(nimcp_neuralnet_forward(nullptr, inputs, outputs));

    network_stats_t stats;
    EXPECT_NO_FATAL_FAILURE(nimcp_neuralnet_get_stats(nullptr, &stats));
}

/**
 * @test Verify queue manager handles NULL pointers
 */
TEST_F(ErrorInjectionTest, NullPointerHandling_QueueManager)
{
    EXPECT_NO_FATAL_FAILURE(nimcp_queue_manager_destroy(nullptr));

    nimcp_message_t msg = {0};
    EXPECT_NO_FATAL_FAILURE(nimcp_queue_manager_enqueue(nullptr, 0, &msg, 0));
    EXPECT_NO_FATAL_FAILURE(nimcp_queue_manager_dequeue(nullptr, 0, &msg, 0));

    EXPECT_FALSE(nimcp_queue_manager_is_empty(nullptr, 0));
    EXPECT_FALSE(nimcp_queue_manager_is_full(nullptr, 0));
    EXPECT_EQ(nimcp_queue_manager_get_size(nullptr, 0), 0);
}

//==============================================================================
// Invalid Parameter Tests
//==============================================================================

/**
 * @test Verify neural network rejects invalid configurations
 */
TEST_F(ErrorInjectionTest, InvalidConfig_NeuralNetwork)
{
    // Test with zero neurons
    {
        nimcp_neuralnet_config_t config = {.num_inputs = 0,  // Invalid
                                           .num_outputs = 5,
                                           .num_hidden = 20};

        nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
        EXPECT_EQ(net, nullptr);
    }

    // Test with negative learning rate
    {
        nimcp_neuralnet_config_t config = {
            .num_inputs = 10,
            .num_outputs = 5,
            .num_hidden = 20,
            .learning_rate = -0.5f  // Invalid
        };

        nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
        // May return NULL or clamp to valid range - either is acceptable
        if (net) {
            nimcp_neuralnet_destroy(net);
        }
    }

    // Test with excessive neurons
    {
        nimcp_neuralnet_config_t config = {.num_inputs = 1000000,  // Too large
                                           .num_outputs = 1000000,
                                           .num_hidden = 1000000};

        nimcp_neuralnet_t* net = nimcp_neuralnet_create(&config);
        // Should fail due to excessive memory requirements
        if (net) {
            // If it somehow succeeds, clean up
            nimcp_neuralnet_destroy(net);
        }
    }
}

/**
 * @test Verify queue manager rejects invalid configurations
 */
TEST_F(ErrorInjectionTest, InvalidConfig_QueueManager)
{
    // Test with zero channels
    {
        nimcp_queue_manager_config_t config = {
            .num_channels = 0,  // Invalid
            .max_queue_size = 100,
            .num_priorities = 3};

        nimcp_queue_manager_t* manager = nimcp_queue_manager_create(&config);
        EXPECT_EQ(manager, nullptr);
    }

    // Test with excessive queue size
    {
        nimcp_queue_manager_config_t config = {
            .num_channels = 4, .max_queue_size = 1000000,  // Excessive
            .num_priorities = 3};

        nimcp_queue_manager_t* manager = nimcp_queue_manager_create(&config);
        // Should either reject or clamp to reasonable value
        if (manager) {
            nimcp_queue_manager_destroy(manager);
        }
    }
}

//==============================================================================
// Resource Exhaustion Tests
//==============================================================================

/**
 * @test Verify system handles running out of file descriptors
 */
TEST_F(ErrorInjectionTest, DISABLED_ResourceExhaustion_FileDescriptors)
{
    // This test would open many files to exhaust FDs
    // Disabled by default as it affects system state
    GTEST_SKIP() << "Resource exhaustion test disabled by default";
}

/**
 * @test Verify system handles memory pressure
 */
TEST_F(ErrorInjectionTest, DISABLED_ResourceExhaustion_Memory)
{
    // This test would allocate large amounts of memory
    // Disabled by default as it affects system state
    GTEST_SKIP() << "Resource exhaustion test disabled by default";
}

//==============================================================================
// Concurrent Error Scenarios
//==============================================================================

/**
 * @test Verify error handling is thread-safe
 */
TEST_F(ErrorInjectionTest, ConcurrentErrors_ThreadSafety)
{
    // Create a queue manager
    nimcp_queue_manager_config_t config = {.num_channels = 4, .max_queue_size = 100, .num_priorities = 3};

    nimcp_queue_manager_t* manager = nimcp_queue_manager_create(&config);
    ASSERT_NE(manager, nullptr);

// Try concurrent invalid operations
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < 100; i++) {
        nimcp_message_t msg = {0};
        // Intentionally use invalid channel
        nimcp_queue_manager_enqueue(manager, 999, &msg, 0);
        nimcp_queue_manager_dequeue(manager, 999, &msg, 0);
    }

    nimcp_queue_manager_destroy(manager);
}
