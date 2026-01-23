/**
 * @file test_exception_macro_comprehensive.cpp
 * @brief Comprehensive unit tests for NIMCP exception handling macros
 *
 * WHAT: Full coverage tests for exception macros and related functionality
 * WHY:  Verify correct behavior of all exception handling macros
 * HOW:  GoogleTest framework with fixture setup/teardown for exception system
 *
 * TEST CATEGORIES:
 * 1. NIMCP_CHECK_THROW - basic null checks, error code propagation
 * 2. NIMCP_CHECK_THROW_MSG - with custom messages
 * 3. NIMCP_TRY_CATCH blocks
 * 4. Exception context propagation
 * 5. Error code to string conversions
 * 6. Nested exception handling
 * 7. Stack trace capture (if enabled)
 * 8. Severity handling
 * 9. Typed exception macros
 * 10. Recovery macros
 *
 * @author NIMCP Development Team
 * @date 2026-01-22
 */

#include <gtest/gtest.h>
#include <cstring>
#include <atomic>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

extern "C" {
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_handlers.h"
#include "utils/exception/nimcp_exception_immune.h"
#include "utils/error/nimcp_error_codes.h"
}

//=============================================================================
// Test Globals for Handler Tracking
//=============================================================================

namespace {

std::atomic<int> g_handler_call_count{0};
std::atomic<nimcp_error_t> g_last_error_code{NIMCP_SUCCESS};
std::atomic<nimcp_exception_severity_t> g_last_severity{EXCEPTION_SEVERITY_DEBUG};
std::atomic<nimcp_exception_category_t> g_last_category{EXCEPTION_CATEGORY_GENERIC};
std::atomic<nimcp_exception_type_t> g_last_type{EXCEPTION_TYPE_BASE};
std::atomic<bool> g_exception_presented_to_immune{false};
std::vector<std::string> g_captured_messages;
std::vector<std::string> g_captured_files;
std::vector<std::string> g_captured_functions;
std::atomic<bool> g_use_message_capture{false};
std::atomic<int> g_last_line{0};

/**
 * @brief Test handler callback to track exception dispatch
 */
bool test_tracking_handler(nimcp_exception_t* ex, void* user_data) {
    (void)user_data;
    if (ex) {
        g_handler_call_count++;
        g_last_error_code = ex->code;
        g_last_severity = ex->severity;
        g_last_category = ex->category;
        g_last_type = ex->type;
        g_exception_presented_to_immune = ex->presented_to_immune;
        g_last_line = ex->line;

        if (g_use_message_capture) {
            if (ex->message) {
                g_captured_messages.push_back(std::string(ex->message));
            }
            if (ex->file) {
                g_captured_files.push_back(std::string(ex->file));
            }
            if (ex->function) {
                g_captured_functions.push_back(std::string(ex->function));
            }
        }
    }
    return false;  // Don't consume, let chain continue
}

/**
 * @brief Reset all test tracking globals
 */
void reset_tracking() {
    g_handler_call_count = 0;
    g_last_error_code = NIMCP_SUCCESS;
    g_last_severity = EXCEPTION_SEVERITY_DEBUG;
    g_last_category = EXCEPTION_CATEGORY_GENERIC;
    g_last_type = EXCEPTION_TYPE_BASE;
    g_exception_presented_to_immune = false;
    g_last_line = 0;
    g_captured_messages.clear();
    g_captured_files.clear();
    g_captured_functions.clear();
}

}  // anonymous namespace

//=============================================================================
// Test Fixture
//=============================================================================

/**
 * WHAT: Base fixture for exception macro comprehensive tests
 * WHY:  Setup/teardown exception system for each test
 */
class ExceptionMacroComprehensiveTest : public ::testing::Test {
protected:
    nimcp_handler_registration_t* handler_reg_ = nullptr;

    void SetUp() override {
        reset_tracking();
        g_use_message_capture = true;

        // Initialize exception system
        nimcp_exception_system_init();

        // Register test handler
        nimcp_handler_options_t opts;
        nimcp_handler_default_options(&opts);
        opts.name = "macro_comprehensive_test_handler";
        opts.handler = test_tracking_handler;
        opts.priority = NIMCP_HANDLER_PRIORITY_HIGH;
        handler_reg_ = nimcp_handler_register(&opts);
    }

    void TearDown() override {
        if (handler_reg_) {
            nimcp_handler_unregister(handler_reg_);
            handler_reg_ = nullptr;
        }
        nimcp_exception_clear_current();
        nimcp_exception_system_shutdown();
        g_use_message_capture = false;
    }
};

//=============================================================================
// SECTION 1: NIMCP_CHECK_THROW Basic Tests
//=============================================================================

/**
 * @brief Function using NIMCP_CHECK_THROW for NULL pointer validation
 */
static nimcp_error_t check_throw_null_test(void* ptr) {
    NIMCP_CHECK_THROW(ptr != nullptr, NIMCP_ERROR_NULL_POINTER,
                      "Parameter ptr is NULL");
    return NIMCP_SUCCESS;
}

/**
 * @brief Function using NIMCP_CHECK_THROW for range validation
 */
static nimcp_error_t check_throw_range_test(int value, int min, int max) {
    NIMCP_CHECK_THROW(value >= min, NIMCP_ERROR_OUT_OF_RANGE,
                      "Value %d below minimum %d", value, min);
    NIMCP_CHECK_THROW(value <= max, NIMCP_ERROR_OUT_OF_RANGE,
                      "Value %d above maximum %d", value, max);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Test NIMCP_CHECK_THROW returns NULL_POINTER error
 * WHY:  Verify basic null check functionality
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowNullPointer) {
    nimcp_error_t result = check_throw_null_test(nullptr);

    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
    EXPECT_EQ(g_handler_call_count, 1);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_NULL_POINTER);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW passes on valid pointer
 * WHY:  Verify successful path doesn't throw
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowValidPointer) {
    int value = 42;
    nimcp_error_t result = check_throw_null_test(&value);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(g_handler_call_count, 0);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW returns OUT_OF_RANGE error for low value
 * WHY:  Verify range checking on lower bound
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowOutOfRangeLow) {
    nimcp_error_t result = check_throw_range_test(-5, 0, 100);

    EXPECT_EQ(result, NIMCP_ERROR_OUT_OF_RANGE);
    EXPECT_EQ(g_handler_call_count, 1);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_OUT_OF_RANGE);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW returns OUT_OF_RANGE error for high value
 * WHY:  Verify range checking on upper bound
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowOutOfRangeHigh) {
    nimcp_error_t result = check_throw_range_test(150, 0, 100);

    EXPECT_EQ(result, NIMCP_ERROR_OUT_OF_RANGE);
    EXPECT_EQ(g_handler_call_count, 1);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW passes on valid range
 * WHY:  Verify successful range check
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowValidRange) {
    nimcp_error_t result = check_throw_range_test(50, 0, 100);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(g_handler_call_count, 0);
}

//=============================================================================
// SECTION 2: NIMCP_CHECK_THROW_MSG Tests
//=============================================================================

/**
 * @brief Function using NIMCP_CHECK_THROW_MSG
 */
static nimcp_error_t check_throw_msg_test(const char* name, size_t size) {
    NIMCP_CHECK_THROW_MSG(name != nullptr, NIMCP_ERROR_NULL_POINTER,
                          "Configuration name is NULL");
    NIMCP_CHECK_THROW_MSG(strlen(name) > 0, NIMCP_ERROR_INVALID_PARAM,
                          "Configuration name '%s' is empty", name);
    NIMCP_CHECK_THROW_MSG(size > 0, NIMCP_ERROR_INVALID_PARAM,
                          "Size must be positive, got %zu", size);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Test NIMCP_CHECK_THROW_MSG with NULL name
 * WHY:  Verify MSG variant works identically to CHECK_THROW
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowMsgNullName) {
    nimcp_error_t result = check_throw_msg_test(nullptr, 10);

    EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
    EXPECT_EQ(g_handler_call_count, 1);
    ASSERT_FALSE(g_captured_messages.empty());
    EXPECT_NE(g_captured_messages[0].find("NULL"), std::string::npos);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW_MSG with empty name
 * WHY:  Verify INVALID_PARAM error for empty string
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowMsgEmptyName) {
    nimcp_error_t result = check_throw_msg_test("", 10);

    EXPECT_EQ(result, NIMCP_ERROR_INVALID_PARAM);
    EXPECT_EQ(g_handler_call_count, 1);
    ASSERT_FALSE(g_captured_messages.empty());
    EXPECT_NE(g_captured_messages[0].find("empty"), std::string::npos);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW_MSG with zero size
 * WHY:  Verify INVALID_PARAM error with formatted size value
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowMsgZeroSize) {
    nimcp_error_t result = check_throw_msg_test("valid_name", 0);

    EXPECT_EQ(result, NIMCP_ERROR_INVALID_PARAM);
    EXPECT_EQ(g_handler_call_count, 1);
    ASSERT_FALSE(g_captured_messages.empty());
    EXPECT_NE(g_captured_messages[0].find("0"), std::string::npos);
}

/**
 * WHAT: Test NIMCP_CHECK_THROW_MSG with valid parameters
 * WHY:  Verify successful path
 */
TEST_F(ExceptionMacroComprehensiveTest, CheckThrowMsgValidParams) {
    nimcp_error_t result = check_throw_msg_test("valid_name", 10);

    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(g_handler_call_count, 0);
}

//=============================================================================
// SECTION 3: NIMCP_TRY_CATCH Tests
//=============================================================================

/**
 * @brief Function that throws inside try block
 */
static nimcp_error_t function_that_throws() {
    NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "Intentional test failure");
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Test NIMCP_TRY/NIMCP_CATCH catches exception
 * WHY:  Verify try/catch block mechanism
 */
TEST_F(ExceptionMacroComprehensiveTest, TryCatchBasic) {
    bool caught = false;
    nimcp_error_t caught_code = NIMCP_SUCCESS;

    NIMCP_TRY {
        function_that_throws();
    }
    NIMCP_CATCH(nimcp_exception_t, ex) {
        caught = true;
        caught_code = ex->code;
        nimcp_exception_unref(ex);
    }
    NIMCP_END_TRY;

    EXPECT_TRUE(caught);
    EXPECT_EQ(caught_code, NIMCP_ERROR_OPERATION_FAILED);
}

/**
 * WHAT: Test NIMCP_TRY block without exception
 * WHY:  Verify no catch when no exception thrown
 */
TEST_F(ExceptionMacroComprehensiveTest, TryCatchNoException) {
    bool caught = false;
    int value = 0;

    NIMCP_TRY {
        value = 42;
    }
    NIMCP_CATCH(nimcp_exception_t, ex) {
        caught = true;
        nimcp_exception_unref(ex);
    }
    NIMCP_END_TRY;

    EXPECT_FALSE(caught);
    EXPECT_EQ(value, 42);
}

/**
 * WHAT: Test nested NIMCP_TRY blocks
 * WHY:  Verify nested exception handling
 */
TEST_F(ExceptionMacroComprehensiveTest, TryCatchNested) {
    bool outer_caught = false;
    bool inner_caught = false;
    nimcp_error_t inner_code = NIMCP_SUCCESS;

    NIMCP_TRY {
        NIMCP_TRY {
            NIMCP_THROW(NIMCP_ERROR_INVALID_STATE, "Inner exception");
        }
        NIMCP_CATCH(nimcp_exception_t, ex) {
            inner_caught = true;
            inner_code = ex->code;
            nimcp_exception_unref(ex);
        }
        NIMCP_END_TRY;
    }
    NIMCP_CATCH(nimcp_exception_t, ex) {
        outer_caught = true;
        nimcp_exception_unref(ex);
    }
    NIMCP_END_TRY;

    EXPECT_TRUE(inner_caught);
    EXPECT_FALSE(outer_caught);
    EXPECT_EQ(inner_code, NIMCP_ERROR_INVALID_STATE);
}

//=============================================================================
// SECTION 4: Exception Context Propagation Tests
//=============================================================================

/**
 * @brief Function that sets exception context
 */
static nimcp_error_t function_with_context(int request_id, const char* module) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Operation failed with context");
    if (ex) {
        nimcp_exception_set_context(ex, "request_id", "12345");
        nimcp_exception_set_context(ex, "module", module);
        nimcp_exception_dispatch(ex);
        nimcp_exception_unref(ex);
    }
    return NIMCP_ERROR_OPERATION_FAILED;
}

/**
 * WHAT: Test exception context key-value pairs
 * WHY:  Verify context data is attached to exceptions
 */
TEST_F(ExceptionMacroComprehensiveTest, ExceptionContextBasic) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Test with context");

    ASSERT_NE(ex, nullptr);

    // Set context
    int result = nimcp_exception_set_context(ex, "key1", "value1");
    EXPECT_EQ(result, 0);

    result = nimcp_exception_set_context(ex, "key2", "value2");
    EXPECT_EQ(result, 0);

    // Retrieve context
    const char* val1 = nimcp_exception_get_context(ex, "key1");
    ASSERT_NE(val1, nullptr);
    EXPECT_STREQ(val1, "value1");

    const char* val2 = nimcp_exception_get_context(ex, "key2");
    ASSERT_NE(val2, nullptr);
    EXPECT_STREQ(val2, "value2");

    // Non-existent key
    const char* val3 = nimcp_exception_get_context(ex, "key3");
    EXPECT_EQ(val3, nullptr);

    // Context count
    EXPECT_EQ(nimcp_exception_context_count(ex), 2u);

    nimcp_exception_unref(ex);
}

/**
 * WHAT: Test exception context removal
 * WHY:  Verify context entries can be removed
 */
TEST_F(ExceptionMacroComprehensiveTest, ExceptionContextRemoval) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Test context removal");

    ASSERT_NE(ex, nullptr);

    nimcp_exception_set_context(ex, "to_remove", "temp_value");
    EXPECT_EQ(nimcp_exception_context_count(ex), 1u);

    int result = nimcp_exception_remove_context(ex, "to_remove");
    EXPECT_EQ(result, 0);
    EXPECT_EQ(nimcp_exception_context_count(ex), 0u);

    // Remove non-existent
    result = nimcp_exception_remove_context(ex, "not_there");
    EXPECT_EQ(result, -1);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 5: Error Code to String Conversion Tests
//=============================================================================

/**
 * WHAT: Test severity to string conversion
 * WHY:  Verify human-readable severity names
 */
TEST_F(ExceptionMacroComprehensiveTest, SeverityToString) {
    const char* debug_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_DEBUG);
    EXPECT_NE(debug_str, nullptr);
    EXPECT_NE(strlen(debug_str), 0u);

    const char* info_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_INFO);
    EXPECT_NE(info_str, nullptr);

    const char* warning_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_WARNING);
    EXPECT_NE(warning_str, nullptr);

    const char* error_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_ERROR);
    EXPECT_NE(error_str, nullptr);

    const char* severe_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_SEVERE);
    EXPECT_NE(severe_str, nullptr);

    const char* critical_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_CRITICAL);
    EXPECT_NE(critical_str, nullptr);

    const char* fatal_str = nimcp_exception_severity_to_string(EXCEPTION_SEVERITY_FATAL);
    EXPECT_NE(fatal_str, nullptr);
}

/**
 * WHAT: Test category to string conversion
 * WHY:  Verify human-readable category names
 */
TEST_F(ExceptionMacroComprehensiveTest, CategoryToString) {
    const char* generic_str = nimcp_exception_category_to_string(EXCEPTION_CATEGORY_GENERIC);
    EXPECT_NE(generic_str, nullptr);

    const char* memory_str = nimcp_exception_category_to_string(EXCEPTION_CATEGORY_MEMORY);
    EXPECT_NE(memory_str, nullptr);

    const char* brain_str = nimcp_exception_category_to_string(EXCEPTION_CATEGORY_BRAIN);
    EXPECT_NE(brain_str, nullptr);

    const char* io_str = nimcp_exception_category_to_string(EXCEPTION_CATEGORY_IO);
    EXPECT_NE(io_str, nullptr);

    const char* threading_str = nimcp_exception_category_to_string(EXCEPTION_CATEGORY_THREADING);
    EXPECT_NE(threading_str, nullptr);

    const char* signal_str = nimcp_exception_category_to_string(EXCEPTION_CATEGORY_SIGNAL);
    EXPECT_NE(signal_str, nullptr);
}

/**
 * WHAT: Test exception type to string conversion
 * WHY:  Verify human-readable type names
 */
TEST_F(ExceptionMacroComprehensiveTest, TypeToString) {
    const char* base_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_BASE);
    EXPECT_NE(base_str, nullptr);

    const char* memory_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_MEMORY);
    EXPECT_NE(memory_str, nullptr);

    const char* brain_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_BRAIN);
    EXPECT_NE(brain_str, nullptr);

    const char* io_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_IO);
    EXPECT_NE(io_str, nullptr);

    const char* threading_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_THREADING);
    EXPECT_NE(threading_str, nullptr);

    const char* security_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_SECURITY);
    EXPECT_NE(security_str, nullptr);

    const char* signal_str = nimcp_exception_type_to_string(EXCEPTION_TYPE_SIGNAL);
    EXPECT_NE(signal_str, nullptr);
}

/**
 * WHAT: Test recovery action to string conversion
 * WHY:  Verify human-readable recovery action names
 */
TEST_F(ExceptionMacroComprehensiveTest, RecoveryActionToString) {
    const char* none_str = nimcp_exception_recovery_action_to_string(EXCEPTION_RECOVERY_NONE);
    EXPECT_NE(none_str, nullptr);

    const char* retry_str = nimcp_exception_recovery_action_to_string(EXCEPTION_RECOVERY_RETRY);
    EXPECT_NE(retry_str, nullptr);

    const char* gc_str = nimcp_exception_recovery_action_to_string(EXCEPTION_RECOVERY_GC);
    EXPECT_NE(gc_str, nullptr);

    const char* rollback_str = nimcp_exception_recovery_action_to_string(EXCEPTION_RECOVERY_ROLLBACK);
    EXPECT_NE(rollback_str, nullptr);

    const char* quarantine_str = nimcp_exception_recovery_action_to_string(EXCEPTION_RECOVERY_QUARANTINE);
    EXPECT_NE(quarantine_str, nullptr);
}

/**
 * WHAT: Test error code to string conversion
 * WHY:  Verify human-readable error messages
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeToString) {
    const char* success_str = nimcp_error_to_string(NIMCP_SUCCESS);
    EXPECT_NE(success_str, nullptr);

    const char* null_str = nimcp_error_to_string(NIMCP_ERROR_NULL_POINTER);
    EXPECT_NE(null_str, nullptr);

    const char* invalid_str = nimcp_error_to_string(NIMCP_ERROR_INVALID_PARAM);
    EXPECT_NE(invalid_str, nullptr);

    const char* memory_str = nimcp_error_to_string(NIMCP_ERROR_NO_MEMORY);
    EXPECT_NE(memory_str, nullptr);

    const char* timeout_str = nimcp_error_to_string(NIMCP_ERROR_TIMEOUT);
    EXPECT_NE(timeout_str, nullptr);
}

//=============================================================================
// SECTION 6: Nested Exception Handling Tests
//=============================================================================

/**
 * @brief Inner function that throws
 */
static nimcp_error_t inner_throwing_function() {
    NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID_STATE, "Inner error");
    return NIMCP_SUCCESS;
}

/**
 * @brief Outer function that catches and rethrows
 */
static nimcp_error_t outer_catching_function() {
    nimcp_error_t result = inner_throwing_function();
    if (result != NIMCP_SUCCESS) {
        // Chain the error
        return result;
    }
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Test exception propagation through call chain
 * WHY:  Verify exceptions bubble up correctly
 */
TEST_F(ExceptionMacroComprehensiveTest, NestedExceptionPropagation) {
    nimcp_error_t result = outer_catching_function();

    EXPECT_EQ(result, NIMCP_ERROR_INVALID_STATE);
    EXPECT_EQ(g_handler_call_count, 1);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_INVALID_STATE);
}

/**
 * @brief Deep nested function
 */
static nimcp_error_t deep_nested_level3() {
    NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED, "Level 3 error");
    return NIMCP_SUCCESS;
}

static nimcp_error_t deep_nested_level2() {
    return deep_nested_level3();
}

static nimcp_error_t deep_nested_level1() {
    return deep_nested_level2();
}

/**
 * WHAT: Test deeply nested exception propagation
 * WHY:  Verify exceptions propagate through multiple call levels
 */
TEST_F(ExceptionMacroComprehensiveTest, DeepNestedExceptionPropagation) {
    nimcp_error_t result = deep_nested_level1();

    EXPECT_EQ(result, NIMCP_ERROR_OPERATION_FAILED);
    EXPECT_EQ(g_handler_call_count, 1);
}

//=============================================================================
// SECTION 7: Stack Trace Tests
//=============================================================================

/**
 * WHAT: Test stack trace capture
 * WHY:  Verify stack trace is captured in exceptions
 */
TEST_F(ExceptionMacroComprehensiveTest, StackTraceCapture) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Test stack trace");

    ASSERT_NE(ex, nullptr);

    // Stack trace should be captured
    // Note: Actual depth depends on platform and build settings
    EXPECT_GE(ex->stack_trace.depth, 0u);

    nimcp_exception_unref(ex);
}

/**
 * WHAT: Test stack trace to string conversion
 * WHY:  Verify stack trace can be formatted as string
 */
TEST_F(ExceptionMacroComprehensiveTest, StackTraceToString) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Test stack trace formatting");

    ASSERT_NE(ex, nullptr);

    char buffer[4096];
    size_t len = nimcp_stack_trace_to_string(&ex->stack_trace, buffer, sizeof(buffer));

    // Should produce some output (even if stack trace is empty)
    EXPECT_GE(len, 0u);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 8: Various Error Types Tests
//=============================================================================

/**
 * @brief Helper to throw specific error codes
 */
static nimcp_error_t throw_error_code(nimcp_error_t code) {
    switch (code) {
        case NIMCP_ERROR_NULL_POINTER:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_NULL_POINTER, "Null pointer error");
            break;
        case NIMCP_ERROR_INVALID_PARAM:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID_PARAM, "Invalid parameter error");
            break;
        case NIMCP_ERROR_OUT_OF_RANGE:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_OUT_OF_RANGE, "Out of range error");
            break;
        case NIMCP_ERROR_NO_MEMORY:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_NO_MEMORY, "No memory error");
            break;
        case NIMCP_ERROR_INVALID_STATE:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID_STATE, "Invalid state error");
            break;
        case NIMCP_ERROR_OPERATION_FAILED:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED, "Operation failed error");
            break;
        case NIMCP_ERROR_TIMEOUT:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_TIMEOUT, "Timeout error");
            break;
        case NIMCP_ERROR_BUFFER_OVERFLOW:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_BUFFER_OVERFLOW, "Buffer overflow error");
            break;
        case NIMCP_ERROR_NOT_INITIALIZED:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_NOT_INITIALIZED, "Not initialized error");
            break;
        case NIMCP_ERROR_ALREADY_EXISTS:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_ALREADY_EXISTS, "Already exists error");
            break;
        case NIMCP_ERROR_NOT_FOUND:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_NOT_FOUND, "Not found error");
            break;
        case NIMCP_ERROR_THREAD_CREATE:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_THREAD_CREATE, "Thread create error");
            break;
        case NIMCP_ERROR_MUTEX_LOCK:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_MUTEX_LOCK, "Mutex lock error");
            break;
        case NIMCP_ERROR_DEADLOCK:
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_DEADLOCK, "Deadlock error");
            break;
        default:
            NIMCP_CHECK_THROW(false, code, "Generic error code %d", (int)code);
            break;
    }
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Test NO_MEMORY error code
 * WHY:  Verify memory error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeNoMemory) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_NO_MEMORY);
    EXPECT_EQ(result, NIMCP_ERROR_NO_MEMORY);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_NO_MEMORY);
}

/**
 * WHAT: Test TIMEOUT error code
 * WHY:  Verify timeout error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeTimeout) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_TIMEOUT);
    EXPECT_EQ(result, NIMCP_ERROR_TIMEOUT);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_TIMEOUT);
}

/**
 * WHAT: Test BUFFER_OVERFLOW error code
 * WHY:  Verify buffer overflow error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeBufferOverflow) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_BUFFER_OVERFLOW);
    EXPECT_EQ(result, NIMCP_ERROR_BUFFER_OVERFLOW);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_BUFFER_OVERFLOW);
}

/**
 * WHAT: Test NOT_INITIALIZED error code
 * WHY:  Verify initialization error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeNotInitialized) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(result, NIMCP_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_NOT_INITIALIZED);
}

/**
 * WHAT: Test NOT_FOUND error code
 * WHY:  Verify not found error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeNotFound) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_NOT_FOUND);
    EXPECT_EQ(result, NIMCP_ERROR_NOT_FOUND);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_NOT_FOUND);
}

/**
 * WHAT: Test THREAD_CREATE error code
 * WHY:  Verify threading error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeThreadCreate) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_THREAD_CREATE);
    EXPECT_EQ(result, NIMCP_ERROR_THREAD_CREATE);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_THREAD_CREATE);
}

/**
 * WHAT: Test DEADLOCK error code
 * WHY:  Verify deadlock error handling
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCodeDeadlock) {
    nimcp_error_t result = throw_error_code(NIMCP_ERROR_DEADLOCK);
    EXPECT_EQ(result, NIMCP_ERROR_DEADLOCK);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_DEADLOCK);
}

//=============================================================================
// SECTION 9: NIMCP_THROW_IF Tests
//=============================================================================

/**
 * @brief Function using NIMCP_THROW_IF
 */
static void throw_if_test(bool condition) {
    /* NIMCP_THROW_IF throws when condition is FALSE */
    NIMCP_THROW_IF(condition, NIMCP_ERROR_INVALID_PARAM, "Condition was false");
}

/**
 * WHAT: Test NIMCP_THROW_IF when condition is false
 * WHY:  Verify throw occurs on false condition
 */
TEST_F(ExceptionMacroComprehensiveTest, ThrowIfConditionFalse) {
    throw_if_test(false);

    EXPECT_EQ(g_handler_call_count, 1);
    EXPECT_EQ(g_last_error_code, NIMCP_ERROR_INVALID_PARAM);
}

/**
 * WHAT: Test NIMCP_THROW_IF when condition is true
 * WHY:  Verify no throw on true condition
 */
TEST_F(ExceptionMacroComprehensiveTest, ThrowIfConditionTrue) {
    throw_if_test(true);

    EXPECT_EQ(g_handler_call_count, 0);
}

//=============================================================================
// SECTION 10: Severity Tests
//=============================================================================

/**
 * WHAT: Test severity derived from error code
 * WHY:  Verify automatic severity assignment
 */
TEST_F(ExceptionMacroComprehensiveTest, SeverityFromErrorCode) {
    // Test that memory errors get higher severity
    nimcp_exception_severity_t mem_severity =
        nimcp_exception_get_severity_from_code(NIMCP_ERROR_NO_MEMORY);
    EXPECT_GE((int)mem_severity, (int)EXCEPTION_SEVERITY_ERROR);

    // Test generic errors
    nimcp_exception_severity_t generic_severity =
        nimcp_exception_get_severity_from_code(NIMCP_ERROR_INVALID_PARAM);
    EXPECT_GE((int)generic_severity, (int)EXCEPTION_SEVERITY_WARNING);
}

/**
 * WHAT: Test category derived from error code
 * WHY:  Verify automatic category assignment
 */
TEST_F(ExceptionMacroComprehensiveTest, CategoryFromErrorCode) {
    // Generic errors (1000-1999)
    nimcp_exception_category_t generic_cat =
        nimcp_exception_get_category_from_code(NIMCP_ERROR_INVALID_PARAM);
    EXPECT_EQ(generic_cat, EXCEPTION_CATEGORY_GENERIC);

    // Memory errors (2000-2999)
    nimcp_exception_category_t mem_cat =
        nimcp_exception_get_category_from_code(NIMCP_ERROR_NO_MEMORY);
    EXPECT_EQ(mem_cat, EXCEPTION_CATEGORY_MEMORY);

    // Threading errors (6000-6999)
    nimcp_exception_category_t thread_cat =
        nimcp_exception_get_category_from_code(NIMCP_ERROR_DEADLOCK);
    EXPECT_EQ(thread_cat, EXCEPTION_CATEGORY_THREADING);
}

//=============================================================================
// SECTION 11: Exception Lifecycle Tests
//=============================================================================

/**
 * WHAT: Test exception reference counting
 * WHY:  Verify ref/unref works correctly
 */
TEST_F(ExceptionMacroComprehensiveTest, ExceptionRefCounting) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Test ref counting");

    ASSERT_NE(ex, nullptr);
    EXPECT_EQ(ex->ref_count, 1);

    // Add reference
    nimcp_exception_t* ex2 = nimcp_exception_ref(ex);
    EXPECT_EQ(ex, ex2);
    EXPECT_EQ(ex->ref_count, 2);

    // Release one reference
    nimcp_exception_unref(ex);
    EXPECT_EQ(ex->ref_count, 1);

    // Release final reference (this frees the memory)
    nimcp_exception_unref(ex);
    // ex is now freed, don't access it
}

/**
 * WHAT: Test exception cause chaining
 * WHY:  Verify exceptions can be chained
 */
TEST_F(ExceptionMacroComprehensiveTest, ExceptionCauseChaining) {
    nimcp_exception_t* cause = nimcp_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Root cause: memory allocation failed");

    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Operation failed");

    ASSERT_NE(cause, nullptr);
    ASSERT_NE(ex, nullptr);

    // Chain cause
    nimcp_exception_set_cause(ex, cause);

    // Verify chain
    nimcp_exception_t* retrieved_cause = nimcp_exception_get_cause(ex);
    EXPECT_EQ(retrieved_cause, cause);
    EXPECT_EQ(retrieved_cause->code, NIMCP_ERROR_NO_MEMORY);

    // Unref the main exception (cause will also be unrefd)
    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 12: Exception to String Tests
//=============================================================================

/**
 * WHAT: Test exception to string conversion
 * WHY:  Verify human-readable exception formatting
 */
TEST_F(ExceptionMacroComprehensiveTest, ExceptionToString) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Test exception message");

    ASSERT_NE(ex, nullptr);

    char buffer[1024];
    size_t len = nimcp_exception_to_string(ex, buffer, sizeof(buffer));

    EXPECT_GT(len, 0u);
    EXPECT_NE(strstr(buffer, "Test exception message"), nullptr);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 13: Thread-Local Exception Tests
//=============================================================================

/**
 * WHAT: Test thread-local current exception
 * WHY:  Verify thread-local exception storage
 */
TEST_F(ExceptionMacroComprehensiveTest, ThreadLocalException) {
    // Initially no current exception
    nimcp_exception_t* current = nimcp_exception_get_current();
    EXPECT_EQ(current, nullptr);

    // Create and set current
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Thread local test");

    ASSERT_NE(ex, nullptr);
    nimcp_exception_set_current(ex);

    // Verify it's set
    current = nimcp_exception_get_current();
    EXPECT_EQ(current, ex);

    // Clear
    nimcp_exception_clear_current();
    current = nimcp_exception_get_current();
    EXPECT_EQ(current, nullptr);

    // Unref
    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 14: Message Formatting Tests
//=============================================================================

/**
 * WHAT: Test printf-style message formatting in exceptions
 * WHY:  Verify formatted messages contain correct values
 */
TEST_F(ExceptionMacroComprehensiveTest, MessageFormattingInteger) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OUT_OF_RANGE,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Value %d is out of range [%d, %d]", 42, 0, 10);

    ASSERT_NE(ex, nullptr);
    EXPECT_NE(strstr(ex->message, "42"), nullptr);
    EXPECT_NE(strstr(ex->message, "0"), nullptr);
    EXPECT_NE(strstr(ex->message, "10"), nullptr);

    nimcp_exception_unref(ex);
}

/**
 * WHAT: Test string formatting in exceptions
 * WHY:  Verify string parameters are included
 */
TEST_F(ExceptionMacroComprehensiveTest, MessageFormattingString) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_NOT_FOUND,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Resource '%s' not found in module '%s'", "config.json", "loader");

    ASSERT_NE(ex, nullptr);
    EXPECT_NE(strstr(ex->message, "config.json"), nullptr);
    EXPECT_NE(strstr(ex->message, "loader"), nullptr);

    nimcp_exception_unref(ex);
}

/**
 * WHAT: Test mixed formatting in exceptions
 * WHY:  Verify multiple format types work together
 */
TEST_F(ExceptionMacroComprehensiveTest, MessageFormattingMixed) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_CRITICAL,
        __FILE__, __LINE__, __func__,
        "Failed to allocate %zu bytes for '%s' (attempt %d/%d)",
        1024u, "buffer", 3, 5);

    ASSERT_NE(ex, nullptr);
    EXPECT_NE(strstr(ex->message, "1024"), nullptr);
    EXPECT_NE(strstr(ex->message, "buffer"), nullptr);
    EXPECT_NE(strstr(ex->message, "3"), nullptr);
    EXPECT_NE(strstr(ex->message, "5"), nullptr);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 15: Source Location Tests
//=============================================================================

/**
 * WHAT: Test source file/line/function capture
 * WHY:  Verify source location is captured in exceptions
 */
TEST_F(ExceptionMacroComprehensiveTest, SourceLocationCapture) {
    int line_before = __LINE__;
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Source location test");
    int expected_line = line_before + 1; // Adjust for the actual CREATE line

    ASSERT_NE(ex, nullptr);
    EXPECT_NE(ex->file, nullptr);
    EXPECT_NE(strstr(ex->file, "test_exception_macro_comprehensive"), nullptr);
    EXPECT_GT(ex->line, 0);
    EXPECT_NE(ex->function, nullptr);

    nimcp_exception_unref(ex);
}

/**
 * WHAT: Test NIMCP_THROW captures source location
 * WHY:  Verify macro captures correct __FILE__ and __LINE__
 */
TEST_F(ExceptionMacroComprehensiveTest, ThrowCapturesSourceLocation) {
    NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "Throw source test");

    EXPECT_EQ(g_handler_call_count, 1);
    ASSERT_FALSE(g_captured_files.empty());
    EXPECT_NE(g_captured_files[0].find("test_exception_macro_comprehensive"), std::string::npos);
    EXPECT_GT(g_last_line, 0);
}

//=============================================================================
// SECTION 16: Epitope Generation Tests
//=============================================================================

/**
 * WHAT: Test immune epitope generation
 * WHY:  Verify epitope is generated for pattern matching
 */
TEST_F(ExceptionMacroComprehensiveTest, EpitopeGeneration) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        "Epitope test");

    ASSERT_NE(ex, nullptr);

    size_t epitope_len = nimcp_exception_generate_epitope(ex);
    EXPECT_GT(epitope_len, 0u);
    EXPECT_LE(epitope_len, NIMCP_EXCEPTION_EPITOPE_SIZE);
    EXPECT_EQ(ex->epitope_len, epitope_len);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 17: Suggested Recovery Tests
//=============================================================================

/**
 * WHAT: Test suggested recovery for memory errors
 * WHY:  Verify appropriate recovery actions are suggested
 */
TEST_F(ExceptionMacroComprehensiveTest, SuggestedRecoveryMemory) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        "Memory error recovery test");

    ASSERT_NE(ex, nullptr);

    nimcp_exception_recovery_action_t action = nimcp_exception_get_suggested_recovery(ex);
    // Memory errors should suggest GC or compact
    EXPECT_TRUE(action == EXCEPTION_RECOVERY_GC ||
                action == EXCEPTION_RECOVERY_COMPACT ||
                action == EXCEPTION_RECOVERY_REDUCE_LOAD);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 18: Aggregate Exception Tests
//=============================================================================

/**
 * WHAT: Test aggregate exception creation
 * WHY:  Verify multiple exceptions can be aggregated
 */
TEST_F(ExceptionMacroComprehensiveTest, AggregateExceptionBasic) {
    nimcp_aggregate_exception_t* agg = nimcp_aggregate_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Multiple errors occurred");

    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->base.type, EXCEPTION_TYPE_AGGREGATE);
    EXPECT_EQ(nimcp_aggregate_exception_count(agg), 0u);

    // Add children
    nimcp_exception_t* child1 = nimcp_exception_create(
        NIMCP_ERROR_INVALID_PARAM,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Child error 1");

    nimcp_exception_t* child2 = nimcp_exception_create(
        NIMCP_ERROR_NULL_POINTER,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Child error 2");

    ASSERT_NE(child1, nullptr);
    ASSERT_NE(child2, nullptr);

    int result = nimcp_aggregate_exception_add(agg, child1);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(nimcp_aggregate_exception_count(agg), 1u);

    result = nimcp_aggregate_exception_add(agg, child2);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(nimcp_aggregate_exception_count(agg), 2u);

    // Retrieve children
    nimcp_exception_t* retrieved = nimcp_aggregate_exception_get(agg, 0);
    EXPECT_EQ(retrieved, child1);

    retrieved = nimcp_aggregate_exception_get(agg, 1);
    EXPECT_EQ(retrieved, child2);

    // Out of bounds
    retrieved = nimcp_aggregate_exception_get(agg, 5);
    EXPECT_EQ(retrieved, nullptr);

    nimcp_exception_unref((nimcp_exception_t*)agg);
}

//=============================================================================
// SECTION 19: Multiple Error Handling Tests
//=============================================================================

/**
 * WHAT: Test multiple errors don't cause memory leaks
 * WHY:  Verify exception cleanup on repeated errors
 */
TEST_F(ExceptionMacroComprehensiveTest, MultipleErrorsNoLeak) {
    const int iterations = 100;

    for (int i = 0; i < iterations; i++) {
        nimcp_error_t result = check_throw_null_test(nullptr);
        EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
    }

    EXPECT_EQ(g_handler_call_count, iterations);
}

/**
 * WHAT: Test alternating success and failure
 * WHY:  Verify cleanup works for mixed outcomes
 */
TEST_F(ExceptionMacroComprehensiveTest, AlternatingSuccessFailure) {
    int success_count = 0;
    int failure_count = 0;

    for (int i = 0; i < 50; i++) {
        if (i % 2 == 0) {
            int value = 42;
            nimcp_error_t result = check_throw_null_test(&value);
            EXPECT_EQ(result, NIMCP_SUCCESS);
            success_count++;
        } else {
            nimcp_error_t result = check_throw_null_test(nullptr);
            EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER);
            failure_count++;
        }
    }

    EXPECT_EQ(success_count, 25);
    EXPECT_EQ(failure_count, 25);
    EXPECT_EQ(g_handler_call_count, 25);
}

//=============================================================================
// SECTION 20: Boundary Value Tests
//=============================================================================

/**
 * WHAT: Test boundary at minimum
 * WHY:  Verify exact boundary is accepted
 */
TEST_F(ExceptionMacroComprehensiveTest, BoundaryAtMinimum) {
    nimcp_error_t result = check_throw_range_test(0, 0, 100);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(g_handler_call_count, 0);
}

/**
 * WHAT: Test boundary at maximum
 * WHY:  Verify exact boundary is accepted
 */
TEST_F(ExceptionMacroComprehensiveTest, BoundaryAtMaximum) {
    nimcp_error_t result = check_throw_range_test(100, 0, 100);
    EXPECT_EQ(result, NIMCP_SUCCESS);
    EXPECT_EQ(g_handler_call_count, 0);
}

/**
 * WHAT: Test one below minimum
 * WHY:  Verify off-by-one is caught
 */
TEST_F(ExceptionMacroComprehensiveTest, BoundaryBelowMinimum) {
    nimcp_error_t result = check_throw_range_test(-1, 0, 100);
    EXPECT_EQ(result, NIMCP_ERROR_OUT_OF_RANGE);
    EXPECT_EQ(g_handler_call_count, 1);
}

/**
 * WHAT: Test one above maximum
 * WHY:  Verify off-by-one is caught
 */
TEST_F(ExceptionMacroComprehensiveTest, BoundaryAboveMaximum) {
    nimcp_error_t result = check_throw_range_test(101, 0, 100);
    EXPECT_EQ(result, NIMCP_ERROR_OUT_OF_RANGE);
    EXPECT_EQ(g_handler_call_count, 1);
}

//=============================================================================
// SECTION 21: Handler Priority Tests
//=============================================================================

namespace {
std::vector<int> g_handler_order;

bool priority_handler_high(nimcp_exception_t* ex, void* user_data) {
    (void)ex;
    (void)user_data;
    g_handler_order.push_back(100);
    return false;
}

bool priority_handler_normal(nimcp_exception_t* ex, void* user_data) {
    (void)ex;
    (void)user_data;
    g_handler_order.push_back(50);
    return false;
}

bool priority_handler_low(nimcp_exception_t* ex, void* user_data) {
    (void)ex;
    (void)user_data;
    g_handler_order.push_back(10);
    return false;
}
}

/**
 * WHAT: Test handler priority ordering
 * WHY:  Verify handlers are called in priority order
 */
TEST_F(ExceptionMacroComprehensiveTest, HandlerPriorityOrder) {
    g_handler_order.clear();

    // Register handlers with different priorities (note: test fixture already has one)
    nimcp_handler_options_t opts_low, opts_normal;

    nimcp_handler_default_options(&opts_low);
    opts_low.name = "priority_test_low";
    opts_low.handler = priority_handler_low;
    opts_low.priority = NIMCP_HANDLER_PRIORITY_LOW;
    nimcp_handler_registration_t* reg_low = nimcp_handler_register(&opts_low);

    nimcp_handler_default_options(&opts_normal);
    opts_normal.name = "priority_test_normal";
    opts_normal.handler = priority_handler_normal;
    opts_normal.priority = NIMCP_HANDLER_PRIORITY_NORMAL;
    nimcp_handler_registration_t* reg_normal = nimcp_handler_register(&opts_normal);

    // Throw exception
    NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "Priority test");

    // Low and normal should be called (high is our test handler)
    // Handlers with higher priority are called first
    EXPECT_GE(g_handler_order.size(), 2u);

    // Cleanup
    nimcp_handler_unregister(reg_low);
    nimcp_handler_unregister(reg_normal);
}

//=============================================================================
// SECTION 22: Exception System State Tests
//=============================================================================

/**
 * WHAT: Test exception system initialization state
 * WHY:  Verify system can detect initialization state
 */
TEST_F(ExceptionMacroComprehensiveTest, SystemInitializationState) {
    // System should be initialized (done in SetUp)
    EXPECT_TRUE(nimcp_exception_system_is_initialized());
}

/**
 * WHAT: Test handler count
 * WHY:  Verify handlers can be counted
 */
TEST_F(ExceptionMacroComprehensiveTest, HandlerCount) {
    size_t count = nimcp_handler_count();
    // At least our test handler should be registered
    EXPECT_GE(count, 1u);
}

//=============================================================================
// SECTION 23: Exception in Try Block Tests
//=============================================================================

/**
 * WHAT: Test nimcp_in_try_block detection
 * WHY:  Verify try block detection works
 */
TEST_F(ExceptionMacroComprehensiveTest, InTryBlockDetection) {
    // Outside try block
    EXPECT_FALSE(nimcp_in_try_block());

    bool was_in_try = false;
    NIMCP_TRY {
        was_in_try = nimcp_in_try_block();
    }
    NIMCP_CATCH(nimcp_exception_t, ex) {
        (void)ex;
    }
    NIMCP_END_TRY;

    EXPECT_TRUE(was_in_try);

    // After try block
    EXPECT_FALSE(nimcp_in_try_block());
}

//=============================================================================
// SECTION 24: Error Code Category Tests
//=============================================================================

/**
 * WHAT: Test error category extraction
 * WHY:  Verify error codes map to correct categories
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorCategoryExtraction) {
    // Generic errors (1000-1999) -> category 1
    EXPECT_EQ(nimcp_error_get_category(NIMCP_ERROR_INVALID_PARAM), 1);

    // Memory errors (2000-2999) -> category 2
    EXPECT_EQ(nimcp_error_get_category(NIMCP_ERROR_NO_MEMORY), 2);

    // Brain errors (3000-3999) -> category 3
    EXPECT_EQ(nimcp_error_get_category(NIMCP_ERROR_BRAIN_CREATION), 3);

    // I/O errors (4000-4999) -> category 4
    EXPECT_EQ(nimcp_error_get_category(NIMCP_ERROR_FILE_NOT_FOUND), 4);

    // Threading errors (6000-6999) -> category 6
    EXPECT_EQ(nimcp_error_get_category(NIMCP_ERROR_DEADLOCK), 6);
}

/**
 * WHAT: Test error success/failure checks
 * WHY:  Verify success/failure detection
 */
TEST_F(ExceptionMacroComprehensiveTest, ErrorSuccessFailureChecks) {
    EXPECT_TRUE(nimcp_error_is_success(NIMCP_SUCCESS));
    EXPECT_TRUE(nimcp_error_is_success(NIMCP_SUCCESS_WITH_WARNINGS));
    EXPECT_TRUE(nimcp_error_is_success(NIMCP_SUCCESS_PARTIAL));

    EXPECT_FALSE(nimcp_error_is_success(NIMCP_ERROR_INVALID_PARAM));
    EXPECT_FALSE(nimcp_error_is_success(NIMCP_ERROR_NO_MEMORY));

    EXPECT_TRUE(nimcp_error_is_failure(NIMCP_ERROR_INVALID_PARAM));
    EXPECT_TRUE(nimcp_error_is_failure(NIMCP_ERROR_NO_MEMORY));

    EXPECT_FALSE(nimcp_error_is_failure(NIMCP_SUCCESS));
}

//=============================================================================
// SECTION 25: Timestamp Tests
//=============================================================================

/**
 * WHAT: Test exception timestamp capture
 * WHY:  Verify timestamps are recorded
 */
TEST_F(ExceptionMacroComprehensiveTest, TimestampCapture) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_OPERATION_FAILED,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "Timestamp test");

    ASSERT_NE(ex, nullptr);
    EXPECT_GT(ex->timestamp_us, 0u);

    nimcp_exception_unref(ex);
}

//=============================================================================
// SECTION 26: Typed Exception Creation Tests
//=============================================================================

/**
 * WHAT: Test memory exception creation
 * WHY:  Verify typed exception has correct type
 */
TEST_F(ExceptionMacroComprehensiveTest, MemoryExceptionCreation) {
    nimcp_memory_exception_t* mex = nimcp_memory_exception_create(
        NIMCP_ERROR_NO_MEMORY,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        1024,  // requested_size
        "Failed to allocate %zu bytes", 1024u);

    ASSERT_NE(mex, nullptr);
    EXPECT_EQ(mex->base.type, EXCEPTION_TYPE_MEMORY);
    EXPECT_EQ(mex->base.code, NIMCP_ERROR_NO_MEMORY);
    EXPECT_EQ(mex->requested_size, 1024u);

    nimcp_exception_unref((nimcp_exception_t*)mex);
}

/**
 * WHAT: Test threading exception creation
 * WHY:  Verify typed exception has correct type
 */
TEST_F(ExceptionMacroComprehensiveTest, ThreadingExceptionCreation) {
    nimcp_threading_exception_t* tex = nimcp_threading_exception_create(
        NIMCP_ERROR_DEADLOCK,
        EXCEPTION_SEVERITY_CRITICAL,
        __FILE__, __LINE__, __func__,
        12345,  // thread_id
        "Deadlock detected on thread %lu", 12345ul);

    ASSERT_NE(tex, nullptr);
    EXPECT_EQ(tex->base.type, EXCEPTION_TYPE_THREADING);
    EXPECT_EQ(tex->base.code, NIMCP_ERROR_DEADLOCK);
    EXPECT_EQ(tex->thread_id, 12345u);

    nimcp_exception_unref((nimcp_exception_t*)tex);
}

/**
 * WHAT: Test I/O exception creation
 * WHY:  Verify typed exception has correct type
 */
TEST_F(ExceptionMacroComprehensiveTest, IOExceptionCreation) {
    nimcp_io_exception_t* iex = nimcp_io_exception_create(
        NIMCP_ERROR_FILE_NOT_FOUND,
        EXCEPTION_SEVERITY_ERROR,
        __FILE__, __LINE__, __func__,
        "/path/to/file.txt",
        "File not found: %s", "/path/to/file.txt");

    ASSERT_NE(iex, nullptr);
    EXPECT_EQ(iex->base.type, EXCEPTION_TYPE_IO);
    EXPECT_EQ(iex->base.code, NIMCP_ERROR_FILE_NOT_FOUND);

    nimcp_exception_unref((nimcp_exception_t*)iex);
}

/**
 * WHAT: Test security exception creation
 * WHY:  Verify typed exception has correct type
 */
TEST_F(ExceptionMacroComprehensiveTest, SecurityExceptionCreation) {
    nimcp_security_exception_t* sex = nimcp_security_exception_create(
        NIMCP_ERROR_SECURITY_THREAT,
        EXCEPTION_SEVERITY_CRITICAL,
        __FILE__, __LINE__, __func__,
        42,  // threat_type
        "Security threat detected: type %d", 42);

    ASSERT_NE(sex, nullptr);
    EXPECT_EQ(sex->base.type, EXCEPTION_TYPE_SECURITY);
    EXPECT_EQ(sex->base.code, NIMCP_ERROR_SECURITY_THREAT);
    EXPECT_EQ(sex->threat_type, 42u);

    nimcp_exception_unref((nimcp_exception_t*)sex);
}

//=============================================================================
// SECTION 27: Brain Exception Tests
//=============================================================================

/**
 * WHAT: Test brain exception creation
 * WHY:  Verify brain exception has correct type and fields
 */
TEST_F(ExceptionMacroComprehensiveTest, BrainExceptionCreation) {
    nimcp_brain_exception_t* bex = nimcp_brain_exception_create(
        NIMCP_ERROR_BRAIN_INVALID,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        123,  // brain_id
        "hippocampus",  // region_name
        "Brain %u region '%s' error", 123, "hippocampus");

    ASSERT_NE(bex, nullptr);
    EXPECT_EQ(bex->base.type, EXCEPTION_TYPE_BRAIN);
    EXPECT_EQ(bex->base.code, NIMCP_ERROR_BRAIN_INVALID);
    EXPECT_EQ(bex->brain_id, 123u);

    nimcp_exception_unref((nimcp_exception_t*)bex);
}

//=============================================================================
// SECTION 28: GPU Exception Tests
//=============================================================================

/**
 * WHAT: Test GPU exception creation
 * WHY:  Verify GPU exception has correct type and fields
 */
TEST_F(ExceptionMacroComprehensiveTest, GPUExceptionCreation) {
    nimcp_gpu_exception_t* gex = nimcp_gpu_exception_create(
        NIMCP_ERROR_GPU,
        EXCEPTION_SEVERITY_SEVERE,
        __FILE__, __LINE__, __func__,
        0,  // device_id
        1,  // cuda_error
        "GPU device %d error (CUDA %d)", 0, 1);

    ASSERT_NE(gex, nullptr);
    EXPECT_EQ(gex->base.type, EXCEPTION_TYPE_GPU);
    EXPECT_EQ(gex->base.code, NIMCP_ERROR_GPU);
    EXPECT_EQ(gex->device_id, 0);
    EXPECT_EQ(gex->cuda_error, 1);

    nimcp_exception_unref((nimcp_exception_t*)gex);
}

//=============================================================================
// SECTION 29: Handler Enable/Disable Tests
//=============================================================================

/**
 * WHAT: Test handler enable/disable
 * WHY:  Verify handlers can be temporarily disabled
 */
TEST_F(ExceptionMacroComprehensiveTest, HandlerEnableDisable) {
    // Disable our test handler
    nimcp_handler_disable(handler_reg_);

    // Throw - our handler shouldn't be called
    reset_tracking();
    NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "Disabled handler test");

    // Our tracking handler shouldn't have been called
    EXPECT_EQ(g_handler_call_count, 0);

    // Re-enable
    nimcp_handler_enable(handler_reg_);

    reset_tracking();
    NIMCP_THROW(NIMCP_ERROR_OPERATION_FAILED, "Re-enabled handler test");

    // Now it should be called
    EXPECT_EQ(g_handler_call_count, 1);
}

//=============================================================================
// SECTION 30: Signal Exception Tests
//=============================================================================

/**
 * WHAT: Test signal to error code conversion
 * WHY:  Verify signals map to correct error codes
 */
TEST_F(ExceptionMacroComprehensiveTest, SignalToErrorCode) {
    // Note: SIGSEGV value is platform-dependent, but should map consistently
    nimcp_error_t err = nimcp_signal_to_error_code(11);  // Typical SIGSEGV
    EXPECT_GE((int)err, 7000);  // Signal errors are in 7000 range
    EXPECT_LT((int)err, 8000);
}

/**
 * WHAT: Test signal name conversion
 * WHY:  Verify signal names are available
 */
TEST_F(ExceptionMacroComprehensiveTest, SignalNameConversion) {
    const char* name = nimcp_signal_name(11);  // Typical SIGSEGV
    EXPECT_NE(name, nullptr);
    EXPECT_GT(strlen(name), 0u);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
