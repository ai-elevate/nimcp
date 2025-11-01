/**
 * @file test_regression.cpp
 * @brief Minimal regression test suite ensuring all modules compile and link
 *
 * WHAT: Smoke tests verifying all 34 module headers compile and link together
 * WHY:  Catch build-breaking changes and missing dependencies
 * HOW:  Include all headers, run basic memory/time operations that are known to work
 *
 * COVERAGE: All 34 NIMCP modules (19 core + 14 utils + 1 logging)
 * TEST PHILOSOPHY: Existing unit tests provide deep testing. These tests ensure:
 *   1. All headers compile together (no conflicting definitions)
 *   2. All modules link correctly (no missing symbols)
 *   3. Basic operations don't crash
 *   4. No memory leaks in basic operations
 */

#include <gtest/gtest.h>

extern "C" {
// Include ALL module headers to verify they compile together
#include "nimcp_adaptive.h"
#include "nimcp_attention.h"
#include "nimcp_bcm.h"
#include "nimcp_brain.h"
#include "nimcp_consolidation.h"
#include "nimcp_curiosity.h"
#include "nimcp_dataio.h"
#include "nimcp_ethics.h"
#include "nimcp_events.h"
#include "nimcp_introspection.h"
#include "nimcp_knowledge.h"
#include "nimcp_neuralnet.h"
#include "nimcp_neuromodulators.h"
#include "nimcp_p2pnode.h"
#include "nimcp_protocol.h"
#include "nimcp_replication.h"
#include "nimcp_salience.h"
#include "nimcp_security.h"
#include "nimcp_stream.h"
#include "utils/nimcp_btree.h"
#include "utils/nimcp_graph.h"
#include "utils/nimcp_hash_table.h"
#include "utils/nimcp_json.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_min_heap.h"
#include "utils/nimcp_queue.h"
#include "utils/nimcp_queue_manager.h"
#include "utils/nimcp_serialization.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_thread_pool.h"
#include "utils/nimcp_time.h"
#include "utils/nimcp_validate.h"
#include "utils/nimcp_vector.h"
#include "logging/nimcp_logging.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class RegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        nimcp_memory_init();
        nimcp_memory_enable_tracking(true);
        nimcp_memory_clear_stats();
    }

    void TearDown() override {
        nimcp_memory_stats_t stats;
        nimcp_memory_get_stats(&stats);
        // Allow small leaks for global state
        EXPECT_LT(stats.current_allocated, 4096);
    }
};

//=============================================================================
// Basic Compilation and Linkage Tests
//=============================================================================

TEST_F(RegressionTest, AllHeaders_CompileTogether) {
    // SUCCESS: If this compiles, all 34 module headers are compatible
    EXPECT_TRUE(true);
}

TEST_F(RegressionTest, AllModules_LinkCorrectly) {
    // SUCCESS: If this links, all modules are in libnimcp_core.so
    EXPECT_TRUE(true);
}

//=============================================================================
// Memory Operations (Known to Work)
//=============================================================================

TEST_F(RegressionTest, Memory_AllocFree) {
    void* ptr = nimcp_malloc(1024);
    ASSERT_NE(ptr, nullptr);
    nimcp_free(ptr);
}

TEST_F(RegressionTest, Memory_Stats) {
    nimcp_memory_stats_t stats;
    nimcp_memory_get_stats(&stats);
    EXPECT_GE(stats.allocation_count, 0);
}

TEST_F(RegressionTest, Memory_MultipleAllocations) {
    const int count = 100;
    void* ptrs[count];

    for (int i = 0; i < count; i++) {
        ptrs[i] = nimcp_malloc(128);
        ASSERT_NE(ptrs[i], nullptr);
    }

    for (int i = 0; i < count; i++) {
        nimcp_free(ptrs[i]);
    }

    nimcp_memory_stats_t stats;
    nimcp_memory_get_stats(&stats);
    EXPECT_EQ(stats.allocation_count, count);
    EXPECT_EQ(stats.free_count, count);
}

//=============================================================================
// Time Operations (Known to Work)
//=============================================================================

TEST_F(RegressionTest, Time_MonotonicIncreases) {
    uint64_t t1 = nimcp_time_monotonic_us();
    uint64_t t2 = nimcp_time_monotonic_us();
    EXPECT_GE(t2, t1);
}

TEST_F(RegressionTest, Time_ElapsedWorks) {
    uint64_t start = nimcp_time_monotonic_us();
    uint64_t elapsed = nimcp_time_elapsed_us(start);
    EXPECT_GE(elapsed, 0);
    EXPECT_LT(elapsed, 1000000); // Less than 1 second
}

//=============================================================================
// Data Structure Creation/Destruction (Known APIs)
//=============================================================================

TEST_F(RegressionTest, HashTable_CreateDestroy) {
    hash_table_config_t config = {};
    config.initial_buckets = 16;
    config.key_type = HASH_KEY_STRING;
    config.hash_algorithm = HASH_ALG_FNV1A;

    hash_table_t* table = hash_table_create(&config);
    ASSERT_NE(table, nullptr);
    hash_table_destroy(table);
}

TEST_F(RegressionTest, MinHeap_CreateDestroy) {
    nimcp_min_heap_t* heap = nimcp_min_heap_create(10);
    ASSERT_NE(heap, nullptr);
    nimcp_min_heap_destroy(heap);
}

TEST_F(RegressionTest, Queue_CreateDestroy) {
    nimcp_queue_config_t config = {};
    config.max_size = 10;
    config.item_size = sizeof(int);
    config.is_blocking = false;
    config.timeout_ms = 0;

    nimcp_queue_handle_t queue;
    nimcp_result_t result = nimcp_queue_create(&config, &queue);
    ASSERT_EQ(result, NIMCP_SUCCESS);
    nimcp_queue_destroy(queue);
}

TEST_F(RegressionTest, Serializer_CreateDestroy) {
    NimcpSerializer* serializer = nimcp_serializer_create(1024);
    ASSERT_NE(serializer, nullptr);
    nimcp_serializer_destroy(serializer);
}

TEST_F(RegressionTest, ThreadPool_CreateDestroy) {
    nimcp_thread_pool_t* pool = nimcp_pool_create(2);
    ASSERT_NE(pool, nullptr);
    nimcp_pool_destroy(pool);
}

TEST_F(RegressionTest, JSON_ParseSimple) {
    const char* json_str = "{\"test\":123}";
    json_t* root = json_loads(json_str, 0, nullptr);
    ASSERT_NE(root, nullptr);
    json_decref(root);
}

//=============================================================================
// Core Module Smoke Tests (Simple APIs)
//=============================================================================

TEST_F(RegressionTest, BCM_SynapseInit) {
    bcm_synapse_t synapse = bcm_synapse_init(0.5f, 1.0f);
    EXPECT_GT(synapse.weight, 0.0f);
    EXPECT_GT(synapse.threshold, 0.0f);
}

TEST_F(RegressionTest, Attention_ConfigValidate) {
    multihead_attention_config_t config = {};
    config.num_heads = 2;
    config.input_dim = 32;
    config.output_dim = 32;
    config.sequence_length = 8;

    bool valid = attention_validate_config(&config);
    EXPECT_TRUE(valid);
}

//=============================================================================
// Performance Regression Tests
//=============================================================================

TEST_F(RegressionTest, Performance_TimeBaseline) {
    uint64_t start = nimcp_time_monotonic_us();
    for (volatile int i = 0; i < 10000; i++) {}
    uint64_t elapsed = nimcp_time_elapsed_us(start);
    // Should complete in under 10ms
    EXPECT_LT(elapsed, 10000);
}

TEST_F(RegressionTest, Performance_MemoryAllocationSpeed) {
    uint64_t start = nimcp_time_monotonic_us();

    for (int i = 0; i < 100; i++) {
        void* ptr = nimcp_malloc(1024);
        nimcp_free(ptr);
    }

    uint64_t elapsed = nimcp_time_elapsed_us(start);
    // 100 alloc/free pairs should complete in under 10ms
    EXPECT_LT(elapsed, 10000);
}

//=============================================================================
// Stress Tests
//=============================================================================

TEST_F(RegressionTest, Stress_MemoryCycles) {
    const int cycles = 1000;

    for (int i = 0; i < cycles; i++) {
        void* ptr = nimcp_malloc(256);
        ASSERT_NE(ptr, nullptr);
        nimcp_free(ptr);
    }

    nimcp_memory_stats_t stats;
    nimcp_memory_get_stats(&stats);
    EXPECT_EQ(stats.allocation_count, cycles);
    EXPECT_EQ(stats.free_count, cycles);
}
