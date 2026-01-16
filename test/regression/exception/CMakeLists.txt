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

# =============================================================================
# Exception API Regression Tests
# Tests that API contracts (enum values, constants, function signatures) remain stable
# =============================================================================

add_executable(regression_exception_test_exception_api_regression
    test_exception_api_regression.cpp
)

target_link_libraries(regression_exception_test_exception_api_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(regression_exception_test_exception_api_regression PRIVATE cxx_std_17)

add_test(
    NAME regression_exception_test_exception_api_regression
    COMMAND regression_exception_test_exception_api_regression
)

set_tests_properties(regression_exception_test_exception_api_regression
    PROPERTIES
    LABELS "regression;exception;api"
    TIMEOUT 120
)

# =============================================================================
# Exception Contracts Regression Tests
# Tests handler registration/unregistration, priority ordering, filtering
# =============================================================================

add_executable(regression_exception_test_exception_contracts_regression
    test_exception_contracts_regression.cpp
)

target_link_libraries(regression_exception_test_exception_contracts_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(regression_exception_test_exception_contracts_regression PRIVATE cxx_std_17)

add_test(
    NAME regression_exception_test_exception_contracts_regression
    COMMAND regression_exception_test_exception_contracts_regression
)

set_tests_properties(regression_exception_test_exception_contracts_regression
    PROPERTIES
    LABELS "regression;exception;contracts;handlers"
    TIMEOUT 120
)

# =============================================================================
# Exception Immune Integration Regression Tests
# Tests exception-immune system integration contracts
# =============================================================================

add_executable(regression_exception_test_exception_immune_regression
    test_exception_immune_regression.cpp
)

target_link_libraries(regression_exception_test_exception_immune_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(regression_exception_test_exception_immune_regression PRIVATE cxx_std_17)

add_test(
    NAME regression_exception_test_exception_immune_regression
    COMMAND regression_exception_test_exception_immune_regression
)

set_tests_properties(regression_exception_test_exception_immune_regression
    PROPERTIES
    LABELS "regression;exception;immune;integration"
    TIMEOUT 120
)
