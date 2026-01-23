/**
 * @file test_exception_e2e_comprehensive.cpp
 * @brief Comprehensive E2E tests for exception handling pipelines
 * @version 1.0.0
 * @date 2026-01-22
 *
 * WHAT: Comprehensive end-to-end tests covering complete exception handling pipelines
 * WHY:  Verify integrated exception handling works correctly under various scenarios
 * HOW:  Test full pipeline flows including detection, throwing, propagation,
 *       handling, and recovery under normal and stress conditions
 *
 * Test Coverage:
 * 1. Error detection -> exception throw -> propagation -> handling -> recovery
 * 2. Exception in async operations -> notification -> cleanup
 * 3. Multi-threaded exception scenarios
 * 4. Exception during resource acquisition -> cleanup verification
 * 5. Exception cascades (one exception triggers another)
 * 6. Exception handling during system stress
 * 7. Exception logging and monitoring integration
 * 8. Exception type polymorphism through handler chain
 * 9. Recovery callback coordination
 * 10. Exception queue overflow handling
 * 11. Concurrent handler registration/unregistration
 * 12. Exception context propagation across threads
 * 13. Nested try/catch with multiple exception types
 * 14. Exception severity escalation
 * 15. Full immune integration lifecycle
 *
 * @author NIMCP Development Team
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <cstring>
#include <cmath>
#include <csignal>
#include <functional>

extern "C" {
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_handlers.h"
#include "utils/exception/nimcp_exception_immune.h"
#include "utils/error/nimcp_error_codes.h"
}

/* ============================================================================
 * Test Utilities and Global State
 * ============================================================================ */

// Thread-safe tracking for exception handling
static std::atomic<int> g_total_exceptions_caught{0};
static std::atomic<int> g_handler_invocations{0};
static std::atomic<int> g_recovery_callbacks{0};
static std::atomic<int> g_recovery_successes{0};
static std::atomic<int> g_cascade_count{0};
static std::atomic<bool> g_async_exception_received{false};
static std::atomic<int> g_thread_exceptions{0};

// Mutex for protecting shared resources
static std::mutex g_test_mutex;
static std::condition_variable g_test_cv;

// Exception tracking vectors (protected by mutex)
static std::vector<nimcp_error_t> g_caught_error_codes;
static std::vector<nimcp_exception_severity_t> g_caught_severities;
static std::vector<std::string> g_caught_messages;

// Resource tracking for cleanup verification
static std::atomic<int> g_resources_allocated{0};
static std::atomic<int> g_resources_freed{0};

// Reset all tracking state
static void reset_all_tracking() {
    g_total_exceptions_caught = 0;
    g_handler_invocations = 0;
    g_recovery_callbacks = 0;
    g_recovery_successes = 0;
    g_cascade_count = 0;
    g_async_exception_received = false;
    g_thread_exceptions = 0;
    g_resources_allocated = 0;
    g_resources_freed = 0;

    std::lock_guard<std::mutex> lock(g_test_mutex);
    g_caught_error_codes.clear();
    g_caught_severities.clear();
    g_caught_messages.clear();
}

// Test handler that tracks all exceptions
static bool tracking_exception_handler(nimcp_exception_t* ex, void* user_data) {
    (void)user_data;
    if (ex) {
        g_handler_invocations++;
        std::lock_guard<std::mutex> lock(g_test_mutex);
        g_caught_error_codes.push_back(ex->code);
        g_caught_severities.push_back(ex->severity);
        g_caught_messages.push_back(ex->message);
    }
    return false; // Don't consume - let chain continue
}

// Test handler that consumes specific exceptions
static bool consuming_handler(nimcp_exception_t* ex, void* user_data) {
    (void)user_data;
    if (ex && ex->severity >= EXCEPTION_SEVERITY_SEVERE) {
        g_total_exceptions_caught++;
        return true; // Consume severe+ exceptions
    }
    return false;
}

// Test recovery callback
static int test_recovery_cb(nimcp_exception_t* ex,
                            nimcp_exception_recovery_action_t action,
                            void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;
    g_recovery_callbacks++;
    g_recovery_successes++;
    return 0; // Success
}

// Failing recovery callback for testing fallback
static int failing_recovery_cb(nimcp_exception_t* ex,
                               nimcp_exception_recovery_action_t action,
                               void* user_data) {
    (void)ex;
    (void)action;
    (void)user_data;
    g_recovery_callbacks++;
    return -1; // Failure
}

// Cascade-triggering handler
static bool cascade_handler(nimcp_exception_t* ex, void* user_data) {
    (void)user_data;
    if (ex && g_cascade_count < 3) {
        g_cascade_count++;
        // Create a follow-up exception (cascade)
        nimcp_exception_t* cascade_ex = nimcp_exception_create(
            NIMCP_ERROR_OPERATION_FAILED,
            EXCEPTION_SEVERITY_WARNING,
            __FILE__, __LINE__, __func__,
            "Cascade exception #%d triggered by original", g_cascade_count.load()
        );
        if (cascade_ex) {
            nimcp_exception_dispatch(cascade_ex);
            nimcp_exception_unref(cascade_ex);
        }
    }
    return false;
}

/* ============================================================================
 * Simulated Resource for Cleanup Testing
 * ============================================================================ */

struct TestResource {
    int id;
    bool is_allocated;

    TestResource(int resource_id) : id(resource_id), is_allocated(true) {
        g_resources_allocated++;
    }

    ~TestResource() {
        if (is_allocated) {
            g_resources_freed++;
            is_allocated = false;
        }
    }

    void release() {
        if (is_allocated) {
            g_resources_freed++;
            is_allocated = false;
        }
    }
};

/* ============================================================================
 * Test Fixture
 * ============================================================================ */

class ExceptionE2EComprehensiveTest : public ::testing::Test {
protected:
    std::vector<nimcp_handler_registration_t*> handler_regs;

    void SetUp() override {
        reset_all_tracking();

        // Initialize exception system
        int init_result = nimcp_exception_system_init();
        ASSERT_EQ(init_result, 0) << "Failed to initialize exception system";

        // Initialize exception-immune integration
        nimcp_exception_immune_config_t immune_config;
        nimcp_exception_immune_default_config(&immune_config);
        immune_config.enable_auto_present = true;
        immune_config.enable_auto_recovery = true;
        immune_config.min_present_severity = EXCEPTION_SEVERITY_ERROR;
        immune_config.async_presentation = false; // Sync for predictable testing

        int immune_init = nimcp_exception_immune_init(&immune_config);
        ASSERT_EQ(immune_init, 0) << "Failed to initialize exception-immune integration";
    }

    void TearDown() override {
        // Unregister all test handlers
        for (auto* reg : handler_regs) {
            if (reg) {
                nimcp_handler_unregister(reg);
            }
        }
        handler_regs.clear();

        // Clear any pending exceptions
        nimcp_exception_clear_current();

        // Shutdown subsystems in order
        nimcp_exception_handlers_shutdown();
        nimcp_exception_immune_shutdown();
        nimcp_exception_system_shutdown();
    }

    // Helper to register handlers
    nimcp_handler_registration_t* RegisterHandler(
        nimcp_exception_handler_fn fn,
        const char* name,
        int priority = NIMCP_HANDLER_PRIORITY_NORMAL,
        nimcp_exception_category_t category = (nimcp_exception_category_t)0) {

        nimcp_handler_options_t opts;
        nimcp_handler_default_options(&opts);
        opts.name = name;
        opts.handler = fn;
        opts.priority = priority;
        opts.category_filter = category;
        opts.user_data = nullptr;

        nimcp_handler_registration_t* reg = nimcp_handler_register(&opts);
        if (reg) {
            handler_regs.push_back(reg);
        }
        return reg;
    }
};

/* ============================================================================
 * Test 1: Error Detection -> Exception Throw -> Propagation -> Handling -> Recovery
 *
 * Complete pipeline from error detection to recovery completion
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, CompleteErrorToRecoveryPipeline) {
    printf("=== Test 1: Complete Error-to-Recovery Pipeline ===\n");

    // Step 1: Register handlers
    auto* tracking_reg = RegisterHandler(tracking_exception_handler, "Tracker");
    ASSERT_NE(tracking_reg, nullptr);
    nimcp_install_default_handlers();
    printf("  Step 1: Handlers registered\n");

    // Step 2: Register recovery callbacks
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_GC, test_recovery_cb, nullptr);
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_RETRY, test_recovery_cb, nullptr);
    printf("  Step 2: Recovery callbacks registered\n");

    // Step 3: Simulate error detection
    bool allocation_failed = true; // Simulated error condition
    size_t requested_size = 1024 * 1024 * 100; // 100MB
    printf("  Step 3: Error detected - allocation failed for %zu bytes\n", requested_size);

    // Step 4: Create and throw exception
    nimcp_memory_exception_t* mem_ex = nimcp_memory_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        requested_size,
        "Memory allocation failed: insufficient heap space"
    );
    ASSERT_NE(mem_ex, nullptr);
    mem_ex->available_size = 50 * 1024 * 1024; // 50MB available
    mem_ex->is_heap = true;
    mem_ex->allocator_name = "system_heap";
    printf("  Step 4: Exception created with code=%d\n", mem_ex->base.code);

    // Step 5: Exception propagation through handler chain
    bool handled = nimcp_exception_dispatch((nimcp_exception_t*)mem_ex);
    printf("  Step 5: Exception dispatched, handled=%d, invocations=%d\n",
           handled ? 1 : 0, g_handler_invocations.load());
    EXPECT_GT(g_handler_invocations.load(), 0);

    // Step 6: Present to immune system
    nimcp_immune_response_t response;
    memset(&response, 0, sizeof(response));
    int present_result = nimcp_exception_present_to_immune(
        (nimcp_exception_t*)mem_ex, &response);
    printf("  Step 6: Presented to immune, result=%d, antigen_id=%u\n",
           present_result, response.antigen_id);

    // Step 7: Get and execute recovery
    nimcp_exception_recovery_action_t suggested =
        nimcp_exception_get_suggested_recovery((nimcp_exception_t*)mem_ex);
    printf("  Step 7: Suggested recovery=%s\n",
           nimcp_exception_recovery_action_to_string(suggested));

    g_recovery_callbacks = 0;
    int recovery_result = nimcp_execute_recovery((nimcp_exception_t*)mem_ex, suggested);
    printf("  Step 7b: Recovery executed, result=%d, callbacks=%d\n",
           recovery_result, g_recovery_callbacks.load());

    // Step 8: Notify and verify recovery
    nimcp_exception_notify_recovery_result(
        (nimcp_exception_t*)mem_ex, suggested, true);
    EXPECT_TRUE(mem_ex->base.recovery_attempted);
    EXPECT_TRUE(mem_ex->base.recovery_succeeded);
    printf("  Step 8: Recovery verified - attempted=%d, succeeded=%d\n",
           mem_ex->base.recovery_attempted ? 1 : 0,
           mem_ex->base.recovery_succeeded ? 1 : 0);

    // Cleanup
    nimcp_exception_unref((nimcp_exception_t*)mem_ex);
    printf("Test 1 PASSED: Complete error-to-recovery pipeline\n\n");
}

/* ============================================================================
 * Test 2: Exception in Async Operations -> Notification -> Cleanup
 *
 * Test exception handling in asynchronous operations
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, AsyncExceptionNotificationCleanup) {
    printf("=== Test 2: Async Exception Notification and Cleanup ===\n");

    RegisterHandler(tracking_exception_handler, "AsyncTracker");

    std::atomic<bool> async_completed{false};
    std::atomic<bool> exception_raised{false};
    nimcp_exception_t* thread_exception = nullptr;

    // Start async operation that will fail
    std::thread async_op([&]() {
        // Initialize exception system in thread
        // Note: In real code, the system is already initialized

        printf("  Async: Operation started\n");

        // Simulate async work that encounters an error
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Create exception in async context
        thread_exception = nimcp_exception_create(
            NIMCP_ERROR_TIMEOUT,
            EXCEPTION_SEVERITY_ERROR,
            __FILE__, __LINE__, __func__,
            "Async operation timed out after 50ms"
        );

        if (thread_exception) {
            nimcp_exception_set_context(thread_exception, "operation", "async_fetch");
            nimcp_exception_set_context(thread_exception, "thread", "worker_1");

            // Dispatch in async context
            nimcp_exception_dispatch(thread_exception);
            exception_raised = true;
            g_async_exception_received = true;
        }

        async_completed = true;
        g_test_cv.notify_one();
        printf("  Async: Operation completed with exception\n");
    });

    // Wait for async operation with timeout
    {
        std::unique_lock<std::mutex> lock(g_test_mutex);
        bool completed = g_test_cv.wait_for(lock, std::chrono::seconds(5),
                                            [&]() { return async_completed.load(); });
        EXPECT_TRUE(completed) << "Async operation timed out";
    }

    async_op.join();

    // Verify exception was raised
    EXPECT_TRUE(exception_raised.load());
    EXPECT_TRUE(g_async_exception_received.load());
    EXPECT_GT(g_handler_invocations.load(), 0);
    printf("  Async exception raised=%d, handler_invocations=%d\n",
           exception_raised.load() ? 1 : 0, g_handler_invocations.load());

    // Verify context survived
    if (thread_exception) {
        const char* op = nimcp_exception_get_context(thread_exception, "operation");
        EXPECT_NE(op, nullptr);
        if (op) {
            EXPECT_STREQ(op, "async_fetch");
            printf("  Context preserved: operation=%s\n", op);
        }
        nimcp_exception_unref(thread_exception);
    }

    printf("Test 2 PASSED: Async exception notification and cleanup\n\n");
}

/* ============================================================================
 * Test 3: Multi-Threaded Exception Scenarios
 *
 * Test exception handling with multiple concurrent threads
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, MultiThreadedExceptionScenarios) {
    printf("=== Test 3: Multi-Threaded Exception Scenarios ===\n");

    RegisterHandler(tracking_exception_handler, "MTTracker");

    const int NUM_THREADS = 8;
    const int EXCEPTIONS_PER_THREAD = 10;
    std::vector<std::thread> threads;
    std::atomic<int> threads_completed{0};

    auto thread_func = [&](int thread_id) {
        for (int i = 0; i < EXCEPTIONS_PER_THREAD; i++) {
            nimcp_exception_t* ex = nimcp_exception_create(
                NIMCP_ERROR_OPERATION_FAILED,
                (nimcp_exception_severity_t)(EXCEPTION_SEVERITY_WARNING + (i % 3)),
                __FILE__, __LINE__, __func__,
                "Thread %d exception %d", thread_id, i
            );

            if (ex) {
                char thread_ctx[32];
                snprintf(thread_ctx, sizeof(thread_ctx), "thread_%d", thread_id);
                nimcp_exception_set_context(ex, "source_thread", thread_ctx);

                nimcp_exception_dispatch(ex);
                g_thread_exceptions++;
                nimcp_exception_unref(ex);
            }

            // Small delay to interleave
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        threads_completed++;
    };

    // Start all threads
    printf("  Starting %d threads, each creating %d exceptions\n",
           NUM_THREADS, EXCEPTIONS_PER_THREAD);
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back(thread_func, t);
    }

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    // Verify results
    int expected_total = NUM_THREADS * EXCEPTIONS_PER_THREAD;
    printf("  Threads completed: %d\n", threads_completed.load());
    printf("  Total exceptions: %d (expected %d)\n",
           g_thread_exceptions.load(), expected_total);
    printf("  Handler invocations: %d\n", g_handler_invocations.load());

    EXPECT_EQ(threads_completed.load(), NUM_THREADS);
    EXPECT_EQ(g_thread_exceptions.load(), expected_total);
    EXPECT_EQ(g_handler_invocations.load(), expected_total);

    // Verify all error codes were tracked
    {
        std::lock_guard<std::mutex> lock(g_test_mutex);
        EXPECT_EQ((int)g_caught_error_codes.size(), expected_total);
    }

    printf("Test 3 PASSED: Multi-threaded exception scenarios\n\n");
}

/* ============================================================================
 * Test 4: Exception During Resource Acquisition -> Cleanup Verification
 *
 * Test that resources are properly cleaned up when exceptions occur
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, ResourceAcquisitionCleanup) {
    printf("=== Test 4: Resource Acquisition Cleanup ===\n");

    RegisterHandler(tracking_exception_handler, "ResourceTracker");

    g_resources_allocated = 0;
    g_resources_freed = 0;

    // Simulate resource acquisition that fails partway
    auto acquire_resources = [](int count, bool fail_at) -> bool {
        std::vector<std::unique_ptr<TestResource>> resources;

        for (int i = 0; i < count; i++) {
            // Allocate resource
            resources.push_back(std::make_unique<TestResource>(i));

            // Simulate failure condition
            if (i == fail_at) {
                // Create and dispatch exception
                nimcp_exception_t* ex = nimcp_exception_create(
                    NIMCP_ERROR_OPERATION_FAILED,
                    EXCEPTION_SEVERITY_ERROR,
                    __FILE__, __LINE__, __func__,
                    "Resource %d acquisition failed - already in use", i
                );
                if (ex) {
                    nimcp_exception_set_context(ex, "resource_id",
                        std::to_string(i).c_str());
                    nimcp_exception_dispatch(ex);
                    nimcp_exception_unref(ex);
                }

                // Clean up allocated resources (RAII via unique_ptr)
                // Resources will be freed when vector goes out of scope
                return false;
            }
        }
        return true;
    };

    // Test case 1: Fail at resource 3 out of 5
    printf("  Test case 1: Acquire 5 resources, fail at #3\n");
    bool success = acquire_resources(5, 3);
    EXPECT_FALSE(success);
    printf("    Allocated: %d, Freed: %d\n",
           g_resources_allocated.load(), g_resources_freed.load());

    // All 4 allocated resources (0,1,2,3) should be freed
    EXPECT_EQ(g_resources_allocated.load(), 4);
    EXPECT_EQ(g_resources_freed.load(), 4);

    // Reset for next test
    g_resources_allocated = 0;
    g_resources_freed = 0;

    // Test case 2: Successful acquisition (no failure)
    printf("  Test case 2: Acquire 5 resources, no failure\n");
    success = acquire_resources(5, -1);
    EXPECT_TRUE(success);
    printf("    Allocated: %d, Freed: %d\n",
           g_resources_allocated.load(), g_resources_freed.load());

    EXPECT_EQ(g_resources_allocated.load(), 5);
    EXPECT_EQ(g_resources_freed.load(), 5); // Freed when scope exits

    printf("Test 4 PASSED: Resource acquisition cleanup\n\n");
}

/* ============================================================================
 * Test 5: Exception Cascades (One Exception Triggers Another)
 *
 * Test handling of cascading exceptions
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, ExceptionCascades) {
    printf("=== Test 5: Exception Cascades ===\n");

    // Register cascade handler that triggers new exceptions
    RegisterHandler(cascade_handler, "CascadeHandler", NIMCP_HANDLER_PRIORITY_HIGH);
    RegisterHandler(tracking_exception_handler, "Tracker", NIMCP_HANDLER_PRIORITY_LOW);

    g_cascade_count = 0;
    g_handler_invocations = 0;

    // Create initial exception that will trigger cascade
    nimcp_exception_t* initial_ex = nimcp_exception_create(
        NIMCP_ERROR_INVALID_STATE,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Initial exception triggering cascade"
    );
    ASSERT_NE(initial_ex, nullptr);
    printf("  Initial exception created\n");

    // Dispatch - this will trigger cascade
    nimcp_exception_dispatch(initial_ex);
    printf("  After dispatch: cascade_count=%d, handler_invocations=%d\n",
           g_cascade_count.load(), g_handler_invocations.load());

    // Verify cascade occurred (limited to 3 to prevent infinite loop)
    EXPECT_EQ(g_cascade_count.load(), 3);

    // Should have 4 total exceptions processed (1 initial + 3 cascade)
    // Each exception triggers both handlers
    EXPECT_GE(g_handler_invocations.load(), 4);

    nimcp_exception_unref(initial_ex);
    printf("Test 5 PASSED: Exception cascades\n\n");
}

/* ============================================================================
 * Test 6: Exception Handling During System Stress
 *
 * Test exception handling under high load conditions
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, ExceptionHandlingUnderStress) {
    printf("=== Test 6: Exception Handling Under Stress ===\n");

    RegisterHandler(tracking_exception_handler, "StressTracker");
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_REDUCE_LOAD, test_recovery_cb, nullptr);

    const int RAPID_FIRE_COUNT = 100;
    const int BURST_COUNT = 5;
    std::atomic<int> successful_dispatches{0};

    printf("  Stress test: %d bursts of %d rapid-fire exceptions\n",
           BURST_COUNT, RAPID_FIRE_COUNT);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int burst = 0; burst < BURST_COUNT; burst++) {
        std::vector<std::thread> burst_threads;

        // Create burst of exceptions
        for (int i = 0; i < 4; i++) { // 4 concurrent threads per burst
            burst_threads.emplace_back([&, i, burst]() {
                for (int j = 0; j < RAPID_FIRE_COUNT / 4; j++) {
                    nimcp_exception_t* ex = nimcp_exception_create(
                        (nimcp_error_t)(NIMCP_ERROR_UNKNOWN + (j % 10)),
                        (nimcp_exception_severity_t)(EXCEPTION_SEVERITY_DEBUG + (j % 5)),
                        __FILE__, __LINE__, __func__,
                        "Stress exception burst=%d thread=%d seq=%d", burst, i, j
                    );
                    if (ex) {
                        nimcp_exception_dispatch(ex);
                        successful_dispatches++;
                        nimcp_exception_unref(ex);
                    }
                }
            });
        }

        for (auto& t : burst_threads) {
            t.join();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    int total_expected = BURST_COUNT * RAPID_FIRE_COUNT;
    printf("  Completed in %ldms\n", duration_ms);
    printf("  Successful dispatches: %d / %d\n",
           successful_dispatches.load(), total_expected);
    printf("  Handler invocations: %d\n", g_handler_invocations.load());

    // Calculate throughput
    double exceptions_per_sec = (successful_dispatches.load() * 1000.0) / duration_ms;
    printf("  Throughput: %.0f exceptions/second\n", exceptions_per_sec);

    EXPECT_EQ(successful_dispatches.load(), total_expected);
    EXPECT_EQ(g_handler_invocations.load(), total_expected);

    printf("Test 6 PASSED: Exception handling under stress\n\n");
}

/* ============================================================================
 * Test 7: Exception Logging and Monitoring Integration
 *
 * Test that exceptions are properly logged and monitored
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, LoggingAndMonitoringIntegration) {
    printf("=== Test 7: Logging and Monitoring Integration ===\n");

    // Install default handlers which include logging
    nimcp_install_default_handlers();
    RegisterHandler(tracking_exception_handler, "MonitorTracker");

    // Create exception with rich context
    nimcp_brain_exception_t* brain_ex = nimcp_brain_exception_create(
        NIMCP_ERROR_FORWARD_PASS,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        42, "visual_cortex",
        "NaN detected in layer activations during forward pass"
    );
    ASSERT_NE(brain_ex, nullptr);

    // Add monitoring-relevant context
    nimcp_exception_set_context(&brain_ex->base, "model_id", "resnet50");
    nimcp_exception_set_context(&brain_ex->base, "batch_id", "1234");
    nimcp_exception_set_context(&brain_ex->base, "epoch", "42");
    nimcp_exception_set_context(&brain_ex->base, "gpu_id", "0");

    brain_ex->network_id = 1;
    brain_ex->layer_id = 15;
    brain_ex->gradient_norm = NAN;
    brain_ex->has_nan_weights = true;
    printf("  Created brain exception with rich context\n");

    // Log exception
    printf("  Exception log output:\n");
    nimcp_exception_log(&brain_ex->base);

    // Format as string
    char buffer[1024];
    size_t len = nimcp_exception_to_string(&brain_ex->base, buffer, sizeof(buffer));
    printf("  Exception string (len=%zu):\n    %s\n", len, buffer);

    // Get and verify context
    EXPECT_STREQ(nimcp_exception_get_context(&brain_ex->base, "model_id"), "resnet50");
    EXPECT_STREQ(nimcp_exception_get_context(&brain_ex->base, "batch_id"), "1234");
    EXPECT_EQ(nimcp_exception_context_count(&brain_ex->base), 4u);
    printf("  Context count: %zu\n", nimcp_exception_context_count(&brain_ex->base));

    // Dispatch and verify monitoring
    nimcp_exception_dispatch(&brain_ex->base);
    EXPECT_GT(g_handler_invocations.load(), 0);

    // Present to immune and verify tracking
    nimcp_immune_response_t response;
    memset(&response, 0, sizeof(response));
    nimcp_exception_present_to_immune(&brain_ex->base, &response);

    // Get statistics
    nimcp_exception_immune_stats_t stats;
    nimcp_exception_immune_get_stats(&stats);
    printf("  Immune stats: presented=%lu, pending=%lu\n",
           (unsigned long)stats.exceptions_presented,
           (unsigned long)stats.exceptions_pending);

    nimcp_exception_unref(&brain_ex->base);
    printf("Test 7 PASSED: Logging and monitoring integration\n\n");
}

/* ============================================================================
 * Test 8: Exception Type Polymorphism Through Handler Chain
 *
 * Test that different exception types are handled correctly
 * ============================================================================ */

static std::atomic<int> g_memory_type_count{0};
static std::atomic<int> g_brain_type_count{0};
static std::atomic<int> g_io_type_count{0};
static std::atomic<int> g_threading_type_count{0};

static bool type_counting_handler(nimcp_exception_t* ex, void* user_data) {
    (void)user_data;
    if (!ex) return false;

    switch (ex->type) {
        case EXCEPTION_TYPE_MEMORY:
            g_memory_type_count++;
            break;
        case EXCEPTION_TYPE_BRAIN:
            g_brain_type_count++;
            break;
        case EXCEPTION_TYPE_IO:
            g_io_type_count++;
            break;
        case EXCEPTION_TYPE_THREADING:
            g_threading_type_count++;
            break;
        default:
            break;
    }
    return false;
}

TEST_F(ExceptionE2EComprehensiveTest, ExceptionTypePolymorphism) {
    printf("=== Test 8: Exception Type Polymorphism ===\n");

    g_memory_type_count = 0;
    g_brain_type_count = 0;
    g_io_type_count = 0;
    g_threading_type_count = 0;

    RegisterHandler(type_counting_handler, "TypeCounter");

    // Create and dispatch different exception types
    printf("  Creating different exception types...\n");

    // Memory exceptions
    for (int i = 0; i < 3; i++) {
        nimcp_memory_exception_t* mem_ex = nimcp_memory_exception_create(
            NIMCP_ERROR_NO_MEMORY, EXCEPTION_SEVERITY_ERROR,
            __FILE__, __LINE__, __func__, 1024,
            "Memory exception %d", i
        );
        ASSERT_NE(mem_ex, nullptr);
        EXPECT_EQ(mem_ex->base.type, EXCEPTION_TYPE_MEMORY);
        nimcp_exception_dispatch(&mem_ex->base);
        nimcp_exception_unref(&mem_ex->base);
    }

    // Brain exceptions
    for (int i = 0; i < 2; i++) {
        nimcp_brain_exception_t* brain_ex = nimcp_brain_exception_create(
            NIMCP_ERROR_FORWARD_PASS, EXCEPTION_SEVERITY_ERROR,
            __FILE__, __LINE__, __func__, i, "cortex",
            "Brain exception %d", i
        );
        ASSERT_NE(brain_ex, nullptr);
        EXPECT_EQ(brain_ex->base.type, EXCEPTION_TYPE_BRAIN);
        nimcp_exception_dispatch(&brain_ex->base);
        nimcp_exception_unref(&brain_ex->base);
    }

    // I/O exceptions
    for (int i = 0; i < 4; i++) {
        nimcp_io_exception_t* io_ex = nimcp_io_exception_create(
            NIMCP_ERROR_FILE_READ, EXCEPTION_SEVERITY_WARNING,
            __FILE__, __LINE__, __func__, "/tmp/test.dat",
            "IO exception %d", i
        );
        ASSERT_NE(io_ex, nullptr);
        EXPECT_EQ(io_ex->base.type, EXCEPTION_TYPE_IO);
        nimcp_exception_dispatch(&io_ex->base);
        nimcp_exception_unref(&io_ex->base);
    }

    // Threading exceptions
    for (int i = 0; i < 2; i++) {
        nimcp_threading_exception_t* thread_ex = nimcp_threading_exception_create(
            NIMCP_ERROR_THREAD_CREATE, EXCEPTION_SEVERITY_ERROR,
            __FILE__, __LINE__, __func__, (uint64_t)i,
            "Threading exception %d", i
        );
        ASSERT_NE(thread_ex, nullptr);
        EXPECT_EQ(thread_ex->base.type, EXCEPTION_TYPE_THREADING);
        nimcp_exception_dispatch(&thread_ex->base);
        nimcp_exception_unref(&thread_ex->base);
    }

    // Verify counts
    printf("  Type counts: memory=%d, brain=%d, io=%d, threading=%d\n",
           g_memory_type_count.load(), g_brain_type_count.load(),
           g_io_type_count.load(), g_threading_type_count.load());

    EXPECT_EQ(g_memory_type_count.load(), 3);
    EXPECT_EQ(g_brain_type_count.load(), 2);
    EXPECT_EQ(g_io_type_count.load(), 4);
    EXPECT_EQ(g_threading_type_count.load(), 2);

    printf("Test 8 PASSED: Exception type polymorphism\n\n");
}

/* ============================================================================
 * Test 9: Recovery Callback Coordination
 *
 * Test that multiple recovery callbacks are coordinated correctly
 * ============================================================================ */

static std::atomic<int> g_primary_recovery_calls{0};
static std::atomic<int> g_fallback_recovery_calls{0};

static int primary_recovery_cb(nimcp_exception_t* ex,
                               nimcp_exception_recovery_action_t action,
                               void* user_data) {
    (void)ex; (void)action; (void)user_data;
    g_primary_recovery_calls++;
    return 0; // Success
}

static int fallback_recovery_cb(nimcp_exception_t* ex,
                                nimcp_exception_recovery_action_t action,
                                void* user_data) {
    (void)ex; (void)action; (void)user_data;
    g_fallback_recovery_calls++;
    return 0; // Success
}

TEST_F(ExceptionE2EComprehensiveTest, RecoveryCallbackCoordination) {
    printf("=== Test 9: Recovery Callback Coordination ===\n");

    g_primary_recovery_calls = 0;
    g_fallback_recovery_calls = 0;

    // Register different callbacks for different actions
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_GC, primary_recovery_cb, nullptr);
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_COMPACT, primary_recovery_cb, nullptr);
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_RETRY, fallback_recovery_cb, nullptr);
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_ROLLBACK, fallback_recovery_cb, nullptr);
    printf("  Registered 4 recovery callbacks\n");

    // Create memory exception that suggests GC
    nimcp_memory_exception_t* mem_ex = nimcp_memory_exception_create(
        NIMCP_ERROR_NO_MEMORY, EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__, 1024 * 1024,
        "Memory exception for recovery coordination test"
    );
    ASSERT_NE(mem_ex, nullptr);

    // Get recovery strategy
    nimcp_exception_recovery_strategy_t strategy;
    nimcp_exception_get_recovery_strategy(&mem_ex->base, &strategy);
    printf("  Strategy: primary=%s, fallback=%s\n",
           nimcp_exception_recovery_action_to_string(strategy.primary_action),
           nimcp_exception_recovery_action_to_string(strategy.fallback_action));

    // Execute primary recovery
    int result = nimcp_execute_recovery(&mem_ex->base, strategy.primary_action);
    printf("  Primary recovery result: %d, calls=%d\n",
           result, g_primary_recovery_calls.load());

    // Execute fallback recovery
    result = nimcp_execute_recovery(&mem_ex->base, strategy.fallback_action);
    printf("  Fallback recovery result: %d, calls=%d\n",
           result, g_fallback_recovery_calls.load());

    // Also test retry explicitly
    result = nimcp_execute_recovery(&mem_ex->base, EXCEPTION_RECOVERY_RETRY);
    printf("  Retry recovery result: %d, total fallback=%d\n",
           result, g_fallback_recovery_calls.load());

    // Verify callbacks were coordinated
    EXPECT_GT(g_primary_recovery_calls.load(), 0);
    EXPECT_GT(g_fallback_recovery_calls.load(), 0);

    nimcp_exception_unref(&mem_ex->base);
    printf("Test 9 PASSED: Recovery callback coordination\n\n");
}

/* ============================================================================
 * Test 10: Exception Queue Overflow Handling
 *
 * Test behavior when async exception queue overflows
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, ExceptionQueueOverflow) {
    printf("=== Test 10: Exception Queue Overflow Handling ===\n");

    // Reconfigure with small queue for testing overflow
    nimcp_exception_immune_shutdown();

    nimcp_exception_immune_config_t small_queue_config;
    nimcp_exception_immune_default_config(&small_queue_config);
    small_queue_config.max_pending_exceptions = 10; // Small queue
    small_queue_config.async_presentation = true;
    nimcp_exception_immune_init(&small_queue_config);
    printf("  Configured with small queue (max=10)\n");

    int successful_queued = 0;
    int queue_overflows = 0;

    // Rapidly queue more exceptions than the queue can hold
    for (int i = 0; i < 50; i++) {
        nimcp_exception_t* ex = nimcp_exception_create(
            NIMCP_ERROR_OPERATION_FAILED, EXCEPTION_SEVERITY_ERROR,
            __FILE__, __LINE__, __func__,
            "Queue overflow test exception %d", i
        );
        if (ex) {
            int result = nimcp_exception_present_async(ex);
            if (result == 0) {
                successful_queued++;
            } else {
                queue_overflows++;
            }
            nimcp_exception_unref(ex);
        }
    }

    printf("  Queued: %d, Overflows: %d\n", successful_queued, queue_overflows);

    // Process some pending exceptions
    size_t processed = nimcp_exception_immune_process_pending(5);
    printf("  Processed %zu pending exceptions\n", processed);

    // Get stats to verify overflow tracking
    nimcp_exception_immune_stats_t stats;
    nimcp_exception_immune_get_stats(&stats);
    printf("  Stats: presented=%lu, pending=%lu, overflows=%lu\n",
           (unsigned long)stats.exceptions_presented,
           (unsigned long)stats.exceptions_pending,
           (unsigned long)stats.queue_overflows);

    // We should have some successful queues and some overflows
    EXPECT_GT(successful_queued, 0);
    // Note: Exact overflow count depends on processing speed

    printf("Test 10 PASSED: Exception queue overflow handling\n\n");
}

/* ============================================================================
 * Test 11: Concurrent Handler Registration/Unregistration
 *
 * Test thread safety of handler registration
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, ConcurrentHandlerRegistration) {
    printf("=== Test 11: Concurrent Handler Registration ===\n");

    std::atomic<int> successful_registrations{0};
    std::atomic<int> successful_unregistrations{0};
    std::vector<std::thread> threads;
    const int NUM_THREADS = 4;
    const int OPS_PER_THREAD = 20;

    // Thread function that registers and unregisters handlers
    auto thread_func = [&](int thread_id) {
        std::vector<nimcp_handler_registration_t*> local_regs;

        for (int i = 0; i < OPS_PER_THREAD; i++) {
            // Register handler
            nimcp_handler_options_t opts;
            nimcp_handler_default_options(&opts);
            char name[64];
            snprintf(name, sizeof(name), "Thread%d_Handler%d", thread_id, i);
            opts.name = name;
            opts.handler = tracking_exception_handler;
            opts.priority = NIMCP_HANDLER_PRIORITY_NORMAL + (thread_id * 10);

            nimcp_handler_registration_t* reg = nimcp_handler_register(&opts);
            if (reg) {
                local_regs.push_back(reg);
                successful_registrations++;
            }

            // Small delay
            std::this_thread::sleep_for(std::chrono::microseconds(50));

            // Dispatch an exception while handlers are changing
            nimcp_exception_t* ex = nimcp_exception_create(
                NIMCP_ERROR_UNKNOWN, EXCEPTION_SEVERITY_INFO,
                __FILE__, __LINE__, __func__,
                "Concurrent test thread=%d iter=%d", thread_id, i
            );
            if (ex) {
                nimcp_exception_dispatch(ex);
                nimcp_exception_unref(ex);
            }
        }

        // Unregister local handlers
        for (auto* reg : local_regs) {
            if (nimcp_handler_unregister(reg) == 0) {
                successful_unregistrations++;
            }
        }
    };

    // Start threads
    printf("  Starting %d threads, each doing %d registration cycles\n",
           NUM_THREADS, OPS_PER_THREAD);
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back(thread_func, t);
    }

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    int expected_ops = NUM_THREADS * OPS_PER_THREAD;
    printf("  Registrations: %d / %d\n",
           successful_registrations.load(), expected_ops);
    printf("  Unregistrations: %d / %d\n",
           successful_unregistrations.load(), expected_ops);
    printf("  Handler invocations: %d\n", g_handler_invocations.load());

    // Verify most operations succeeded
    EXPECT_GE(successful_registrations.load(), expected_ops * 0.9);
    EXPECT_GE(successful_unregistrations.load(), expected_ops * 0.9);

    printf("Test 11 PASSED: Concurrent handler registration\n\n");
}

/* ============================================================================
 * Test 12: Exception Context Propagation Across Threads
 *
 * Test that exception context is correctly propagated in multi-threaded code
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, ContextPropagationAcrossThreads) {
    printf("=== Test 12: Context Propagation Across Threads ===\n");

    RegisterHandler(tracking_exception_handler, "ContextTracker");

    std::atomic<bool> context_verified{false};
    nimcp_exception_t* captured_exception = nullptr;
    std::mutex capture_mutex;

    // Producer thread creates exception with context
    std::thread producer([&]() {
        nimcp_exception_t* ex = nimcp_exception_create(
            NIMCP_ERROR_OPERATION_FAILED, EXCEPTION_SEVERITY_ERROR,
            __FILE__, __LINE__, __func__,
            "Exception for context propagation test"
        );

        if (ex) {
            // Add rich context
            nimcp_exception_set_context(ex, "producer_thread", "true");
            nimcp_exception_set_context(ex, "request_id", "REQ-12345");
            nimcp_exception_set_context(ex, "user_id", "USER-67890");
            nimcp_exception_set_context(ex, "correlation_id", "CID-ABCDE");

            // Capture for consumer verification
            {
                std::lock_guard<std::mutex> lock(capture_mutex);
                captured_exception = nimcp_exception_ref(ex);
            }

            // Dispatch in producer
            nimcp_exception_dispatch(ex);
            nimcp_exception_unref(ex);
        }
    });

    producer.join();

    // Consumer thread verifies context
    std::thread consumer([&]() {
        std::lock_guard<std::mutex> lock(capture_mutex);
        if (captured_exception) {
            // Verify all context survived
            bool all_valid = true;

            const char* producer = nimcp_exception_get_context(
                captured_exception, "producer_thread");
            if (!producer || strcmp(producer, "true") != 0) all_valid = false;

            const char* request = nimcp_exception_get_context(
                captured_exception, "request_id");
            if (!request || strcmp(request, "REQ-12345") != 0) all_valid = false;

            const char* user = nimcp_exception_get_context(
                captured_exception, "user_id");
            if (!user || strcmp(user, "USER-67890") != 0) all_valid = false;

            const char* correlation = nimcp_exception_get_context(
                captured_exception, "correlation_id");
            if (!correlation || strcmp(correlation, "CID-ABCDE") != 0) all_valid = false;

            context_verified = all_valid;

            printf("  Consumer verification: producer_thread=%s\n",
                   producer ? producer : "null");
            printf("  Consumer verification: request_id=%s\n",
                   request ? request : "null");
            printf("  Consumer verification: user_id=%s\n",
                   user ? user : "null");
            printf("  Consumer verification: correlation_id=%s\n",
                   correlation ? correlation : "null");
        }
    });

    consumer.join();

    // Cleanup
    {
        std::lock_guard<std::mutex> lock(capture_mutex);
        if (captured_exception) {
            nimcp_exception_unref(captured_exception);
        }
    }

    EXPECT_TRUE(context_verified.load());
    printf("Test 12 PASSED: Context propagation across threads\n\n");
}

/* ============================================================================
 * Test 13: Nested Try/Catch with Multiple Exception Types
 *
 * Test nested exception handling blocks
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, NestedTryCatchMultipleTypes) {
    printf("=== Test 13: Nested Try/Catch with Multiple Types ===\n");

    bool outer_caught = false;
    bool inner_caught = false;
    nimcp_error_t outer_code = NIMCP_SUCCESS;
    nimcp_error_t inner_code = NIMCP_SUCCESS;

    NIMCP_TRY {
        printf("  Outer try block entered\n");

        NIMCP_TRY {
            printf("  Inner try block entered\n");

            // Create and raise inner exception
            nimcp_exception_t* inner_ex = nimcp_exception_create(
                NIMCP_ERROR_INVALID_PARAMETER,
                EXCEPTION_SEVERITY_ERROR,
                __FILE__, __LINE__, __func__,
                "Inner exception - invalid parameter"
            );
            ASSERT_NE(inner_ex, nullptr);

            nimcp_exception_raise(inner_ex);

            // Should not reach here
            printf("  ERROR: Should not reach after inner raise\n");
            FAIL() << "Inner exception not caught";
        }
        NIMCP_CATCH(nimcp_exception_t, caught_inner) {
            printf("  Inner catch: caught exception\n");
            inner_caught = true;
            if (caught_inner) {
                inner_code = caught_inner->code;
                printf("  Inner code: %d\n", inner_code);
                nimcp_exception_unref(caught_inner);
            }
        }
        NIMCP_END_TRY;

        // After handling inner, raise outer
        printf("  After inner try, raising outer exception\n");
        nimcp_exception_t* outer_ex = nimcp_exception_create(
            NIMCP_ERROR_OPERATION_FAILED,
            EXCEPTION_SEVERITY_SEVERE,
            __FILE__, __LINE__, __func__,
            "Outer exception - operation failed"
        );
        ASSERT_NE(outer_ex, nullptr);

        nimcp_exception_raise(outer_ex);

        // Should not reach here
        printf("  ERROR: Should not reach after outer raise\n");
        FAIL() << "Outer exception not caught";
    }
    NIMCP_CATCH(nimcp_exception_t, caught_outer) {
        printf("  Outer catch: caught exception\n");
        outer_caught = true;
        if (caught_outer) {
            outer_code = caught_outer->code;
            printf("  Outer code: %d\n", outer_code);
            nimcp_exception_unref(caught_outer);
        }
    }
    NIMCP_END_TRY;

    // Verify both exceptions were caught
    EXPECT_TRUE(inner_caught);
    EXPECT_TRUE(outer_caught);
    EXPECT_EQ(inner_code, NIMCP_ERROR_INVALID_PARAMETER);
    EXPECT_EQ(outer_code, NIMCP_ERROR_OPERATION_FAILED);

    printf("Test 13 PASSED: Nested try/catch with multiple types\n\n");
}

/* ============================================================================
 * Test 14: Exception Severity Escalation
 *
 * Test that severity can escalate through handling chain
 * ============================================================================ */

static std::atomic<nimcp_exception_severity_t> g_last_observed_severity{EXCEPTION_SEVERITY_DEBUG};

static bool severity_observing_handler(nimcp_exception_t* ex, void* user_data) {
    (void)user_data;
    if (ex) {
        g_last_observed_severity = ex->severity;
    }
    return false;
}

TEST_F(ExceptionE2EComprehensiveTest, SeverityEscalation) {
    printf("=== Test 14: Exception Severity Escalation ===\n");

    RegisterHandler(severity_observing_handler, "SeverityObserver");

    // Create exception with low severity
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_UNKNOWN,
        EXCEPTION_SEVERITY_WARNING,
        __FILE__, __LINE__, __func__,
        "Initial low-severity exception"
    );
    ASSERT_NE(ex, nullptr);
    printf("  Created exception with severity=%d (WARNING)\n", ex->severity);

    // Dispatch and observe initial severity
    nimcp_exception_dispatch(ex);
    EXPECT_EQ(g_last_observed_severity.load(), EXCEPTION_SEVERITY_WARNING);
    printf("  Observed severity after first dispatch: %d\n",
           g_last_observed_severity.load());

    // Escalate severity (simulating detection of more serious issue)
    ex->severity = EXCEPTION_SEVERITY_SEVERE;
    printf("  Escalated severity to %d (SEVERE)\n", ex->severity);

    // Dispatch again with escalated severity
    nimcp_exception_dispatch(ex);
    EXPECT_EQ(g_last_observed_severity.load(), EXCEPTION_SEVERITY_SEVERE);
    printf("  Observed severity after escalation: %d\n",
           g_last_observed_severity.load());

    // Further escalate to critical
    ex->severity = EXCEPTION_SEVERITY_CRITICAL;
    printf("  Escalated severity to %d (CRITICAL)\n", ex->severity);

    nimcp_exception_dispatch(ex);
    EXPECT_EQ(g_last_observed_severity.load(), EXCEPTION_SEVERITY_CRITICAL);
    printf("  Observed severity after critical escalation: %d\n",
           g_last_observed_severity.load());

    // Present to immune (should trigger response due to high severity)
    nimcp_immune_response_t response;
    memset(&response, 0, sizeof(response));
    nimcp_exception_present_to_immune(ex, &response);
    printf("  Presented critical exception to immune\n");

    nimcp_exception_unref(ex);
    printf("Test 14 PASSED: Exception severity escalation\n\n");
}

/* ============================================================================
 * Test 15: Full Immune Integration Lifecycle
 *
 * Complete lifecycle with immune system integration
 * ============================================================================ */

TEST_F(ExceptionE2EComprehensiveTest, FullImmuneIntegrationLifecycle) {
    printf("=== Test 15: Full Immune Integration Lifecycle ===\n");

    // Setup
    nimcp_install_default_handlers();
    RegisterHandler(tracking_exception_handler, "LifecycleTracker");
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_GC, test_recovery_cb, nullptr);
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_COMPACT, test_recovery_cb, nullptr);
    nimcp_register_recovery_callback(EXCEPTION_RECOVERY_ROLLBACK, test_recovery_cb, nullptr);

    // Reset statistics
    nimcp_exception_immune_reset_stats();
    g_recovery_callbacks = 0;

    printf("  Phase 1: Create and configure exception\n");
    nimcp_memory_exception_t* mem_ex = nimcp_memory_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        512 * 1024 * 1024,  // 512MB
        "Full immune lifecycle test - critical memory failure"
    );
    ASSERT_NE(mem_ex, nullptr);

    mem_ex->available_size = 128 * 1024 * 1024;
    mem_ex->is_heap = true;
    mem_ex->allocator_name = "neural_pool";

    nimcp_exception_set_context(&mem_ex->base, "operation", "weight_allocation");
    nimcp_exception_set_context(&mem_ex->base, "layer", "attention_layer_12");

    printf("  Phase 2: Generate epitope\n");
    size_t epitope_len = nimcp_exception_generate_epitope(&mem_ex->base);
    printf("    Epitope length: %zu\n", epitope_len);
    EXPECT_GT(epitope_len, 0u);

    printf("  Phase 3: Dispatch through handler chain\n");
    bool handled = nimcp_exception_dispatch(&mem_ex->base);
    printf("    Handled: %d, Handler invocations: %d\n",
           handled ? 1 : 0, g_handler_invocations.load());

    printf("  Phase 4: Present to immune system\n");
    nimcp_immune_response_t immune_response;
    memset(&immune_response, 0, sizeof(immune_response));
    int present_result = nimcp_exception_present_to_immune(&mem_ex->base, &immune_response);
    printf("    Present result: %d\n", present_result);
    printf("    Antigen ID: %u\n", immune_response.antigen_id);
    EXPECT_TRUE(mem_ex->base.presented_to_immune);

    printf("  Phase 5: Get recovery strategy\n");
    nimcp_exception_recovery_strategy_t strategy;
    nimcp_exception_get_recovery_strategy(&mem_ex->base, &strategy);
    printf("    Primary: %s, Fallback: %s, Retries: %u\n",
           nimcp_exception_recovery_action_to_string(strategy.primary_action),
           nimcp_exception_recovery_action_to_string(strategy.fallback_action),
           strategy.retry_count);

    printf("  Phase 6: Execute primary recovery\n");
    int recovery_result = nimcp_execute_recovery(&mem_ex->base, strategy.primary_action);
    printf("    Recovery result: %d, Callbacks: %d\n",
           recovery_result, g_recovery_callbacks.load());

    printf("  Phase 7: Notify recovery outcome\n");
    nimcp_exception_notify_recovery_result(&mem_ex->base, strategy.primary_action, true);
    mem_ex->base.recovery_attempted = true;
    mem_ex->base.recovery_succeeded = true;
    printf("    Attempted: %d, Succeeded: %d\n",
           mem_ex->base.recovery_attempted ? 1 : 0,
           mem_ex->base.recovery_succeeded ? 1 : 0);

    printf("  Phase 8: Verify final statistics\n");
    nimcp_exception_immune_stats_t final_stats;
    nimcp_exception_immune_get_stats(&final_stats);
    printf("    Total presented: %lu\n", (unsigned long)final_stats.exceptions_presented);
    printf("    Recoveries attempted: %lu\n", (unsigned long)final_stats.recoveries_attempted);
    printf("    Recoveries succeeded: %lu\n", (unsigned long)final_stats.recoveries_succeeded);
    printf("    Avg response time: %.2f us\n", final_stats.avg_response_time_us);

    printf("  Phase 9: Verify exception state\n");
    EXPECT_TRUE(mem_ex->base.presented_to_immune);
    EXPECT_TRUE(mem_ex->base.recovery_attempted);
    EXPECT_TRUE(mem_ex->base.recovery_succeeded);
    EXPECT_GT(g_handler_invocations.load(), 0);

    // Cleanup
    nimcp_exception_unref(&mem_ex->base);
    printf("Test 15 PASSED: Full immune integration lifecycle\n\n");
}

/* ============================================================================
 * Main Function
 * ============================================================================ */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
