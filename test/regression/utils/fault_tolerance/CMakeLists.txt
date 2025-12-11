# Utils Fault Tolerance Regression Tests
# Enhanced Fault Tolerance Module Regression (Phase FT-2: Immune System)

# Helper macro to add regression test
macro(add_ft_regression_test TEST_NAME)
    add_executable(regression_utils_fault_tolerance_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(regression_utils_fault_tolerance_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            pthread
    )
    target_include_directories(regression_utils_fault_tolerance_${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME regression_utils_fault_tolerance_${TEST_NAME}
             COMMAND regression_utils_fault_tolerance_${TEST_NAME})
    set_tests_properties(regression_utils_fault_tolerance_${TEST_NAME} PROPERTIES LABELS "regression")
endmacro()

# Existing regression tests
add_ft_regression_test(test_checkpoint_format)
add_ft_regression_test(test_diagnostics_regression)
add_ft_regression_test(test_fast_recovery_regression)
add_ft_regression_test(test_fault_event_bus_regression)
add_ft_regression_test(test_lockfree_metrics_regression)
add_ft_regression_test(test_recovery_cache_regression)
add_ft_regression_test(test_recovery_pool_regression)

# Enhanced Fault Tolerance Regression Tests (Phase FT-2)
add_ft_regression_test(test_enhanced_fault_tolerance_regression)
