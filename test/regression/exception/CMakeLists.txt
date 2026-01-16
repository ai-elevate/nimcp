# =============================================================================
# Exception Regression Tests
# =============================================================================

# Signal Exception Regression Tests
add_executable(regression_exception_test_signal_exception_regression
    test_signal_exception_regression.cpp
)

target_link_libraries(regression_exception_test_signal_exception_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(regression_exception_test_signal_exception_regression PRIVATE cxx_std_17)

add_test(
    NAME regression_exception_test_signal_exception_regression
    COMMAND regression_exception_test_signal_exception_regression
)

set_tests_properties(regression_exception_test_signal_exception_regression
    PROPERTIES
    LABELS "regression;exception;signal"
    TIMEOUT 120
)
