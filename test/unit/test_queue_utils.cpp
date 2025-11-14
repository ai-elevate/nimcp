/**
 * @file test_queue_utils.cpp
 * @brief Unit tests for queue utility
 */

#include <gtest/gtest.h>
#include "utils/memory/nimcp_memory.h"
#include "utils/containers/nimcp_queue.h"

// Note: queue_manager tests are skipped due to C++/C11 atomics incompatibility
// The queue_manager uses atomic types in its public API which don't work with C++

class QueueTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        nimcp_memory_init();
    }

    void TearDown() override
    {
        nimcp_memory_cleanup();
    }
};

/**
 * WHAT: Test basic queue creation and destruction
 * WHY: Verify queue can be created with valid configuration
 */
TEST_F(QueueTest, CreateDestroy)
{
    nimcp_queue_config_t config;
    config.max_size = 100;
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue = nullptr;
    ASSERT_EQ(nimcp_queue_create(&config, &queue), NIMCP_SUCCESS);
    ASSERT_NE(queue, nullptr);

    ASSERT_EQ(nimcp_queue_destroy(queue), NIMCP_SUCCESS);
}

/**
 * WHAT: Test enqueue and dequeue operations
 * WHY: Verify basic queue functionality
 */
TEST_F(QueueTest, EnqueueDequeue)
{
    nimcp_queue_config_t config;
    config.max_size = 10;
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue = nullptr;
    ASSERT_EQ(nimcp_queue_create(&config, &queue), NIMCP_SUCCESS);

    // Enqueue some items
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(nimcp_queue_enqueue(queue, &i, 0), NIMCP_SUCCESS);
    }

    // Verify size
    ASSERT_EQ(nimcp_queue_get_size(queue), 5);
    ASSERT_FALSE(nimcp_queue_is_empty(queue));

    // Dequeue and verify order
    for (int i = 0; i < 5; i++) {
        int value = -1;
        ASSERT_EQ(nimcp_queue_dequeue(queue, &value, 0), NIMCP_SUCCESS);
        ASSERT_EQ(value, i);
    }

    // Should be empty now
    ASSERT_TRUE(nimcp_queue_is_empty(queue));

    ASSERT_EQ(nimcp_queue_destroy(queue), NIMCP_SUCCESS);
}

/**
 * WHAT: Test queue full condition
 * WHY: Verify queue properly handles overflow
 * NOTE: Circular buffer keeps one slot empty, so max_size=4 holds 3 items
 */
TEST_F(QueueTest, QueueFull)
{
    nimcp_queue_config_t config;
    config.max_size = 4;  // Holds 3 items (circular buffer uses 1 slot for empty detection)
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue = nullptr;
    ASSERT_EQ(nimcp_queue_create(&config, &queue), NIMCP_SUCCESS);

    // Fill the queue (can hold max_size - 1 items)
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(nimcp_queue_enqueue(queue, &i, 0), NIMCP_SUCCESS);
    }

    ASSERT_TRUE(nimcp_queue_is_full(queue));

    // Try to enqueue when full - should fail
    int extra = 999;
    ASSERT_NE(nimcp_queue_enqueue(queue, &extra, 0), NIMCP_SUCCESS);

    ASSERT_EQ(nimcp_queue_destroy(queue), NIMCP_SUCCESS);
}

/**
 * WHAT: Test queue peek operation
 * WHY: Verify we can look at front item without dequeuing
 */
TEST_F(QueueTest, Peek)
{
    nimcp_queue_config_t config;
    config.max_size = 10;
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue = nullptr;
    ASSERT_EQ(nimcp_queue_create(&config, &queue), NIMCP_SUCCESS);

    int value = 42;
    ASSERT_EQ(nimcp_queue_enqueue(queue, &value, 0), NIMCP_SUCCESS);

    // Peek should return the value without removing it
    int peeked = -1;
    ASSERT_EQ(nimcp_queue_peek(queue, &peeked), NIMCP_SUCCESS);
    ASSERT_EQ(peeked, 42);

    // Queue should still have the item
    ASSERT_EQ(nimcp_queue_get_size(queue), 1);

    ASSERT_EQ(nimcp_queue_destroy(queue), NIMCP_SUCCESS);
}

/**
 * WHAT: Test queue clear operation
 * WHY: Verify we can empty a queue at once
 */
TEST_F(QueueTest, Clear)
{
    nimcp_queue_config_t config;
    config.max_size = 10;
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue = nullptr;
    ASSERT_EQ(nimcp_queue_create(&config, &queue), NIMCP_SUCCESS);

    // Add several items
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(nimcp_queue_enqueue(queue, &i, 0), NIMCP_SUCCESS);
    }

    ASSERT_EQ(nimcp_queue_get_size(queue), 5);

    // Clear the queue
    ASSERT_EQ(nimcp_queue_clear(queue), NIMCP_SUCCESS);

    // Should be empty now
    ASSERT_TRUE(nimcp_queue_is_empty(queue));
    ASSERT_EQ(nimcp_queue_get_size(queue), 0);

    ASSERT_EQ(nimcp_queue_destroy(queue), NIMCP_SUCCESS);
}

/**
 * WHAT: Test queue status reporting
 * WHY: Verify statistics are tracked correctly
 */
TEST_F(QueueTest, Status)
{
    nimcp_queue_config_t config;
    config.max_size = 10;
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue = nullptr;
    ASSERT_EQ(nimcp_queue_create(&config, &queue), NIMCP_SUCCESS);

    // Enqueue and dequeue some items
    for (int i = 0; i < 7; i++) {
        ASSERT_EQ(nimcp_queue_enqueue(queue, &i, 0), NIMCP_SUCCESS);
    }

    for (int i = 0; i < 3; i++) {
        int value;
        ASSERT_EQ(nimcp_queue_dequeue(queue, &value, 0), NIMCP_SUCCESS);
    }

    nimcp_queue_status_t status;
    ASSERT_EQ(nimcp_queue_get_status(queue, &status), NIMCP_SUCCESS);

    ASSERT_EQ(status.current_size, 4);
    ASSERT_EQ(status.total_enqueued, 7);
    ASSERT_EQ(status.total_dequeued, 3);

    ASSERT_EQ(nimcp_queue_destroy(queue), NIMCP_SUCCESS);
}
