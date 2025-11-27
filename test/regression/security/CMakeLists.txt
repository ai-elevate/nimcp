# ============================================================================
# Security Regression Tests - Blood-Brain Barrier (Phase IS-1)
# ============================================================================
# WHAT: Regression tests for Neural Immune System backward compatibility
# WHY:  Ensure BBB behavior remains consistent across versions
# HOW:  Test API stability, performance baselines, and security guarantees
# ============================================================================

# Blood-Brain Barrier Regression Tests (32 tests)
add_executable(regression_security_test_bbb_regression
    test_bbb_regression.cpp
)
target_link_libraries(regression_security_test_bbb_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_security_test_bbb_regression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_bbb_regression
         COMMAND regression_security_test_bbb_regression)
set_tests_properties(regression_security_test_bbb_regression PROPERTIES
    LABELS "regression"
    TIMEOUT 60
)

# ============================================================================
# Mathematical Security Regression Tests (Phase SC-3)
# ============================================================================
# WHAT: Regression tests for Entropy, Bayesian Trust, and Differential Privacy
# WHY:  Ensure mathematical correctness and API stability across versions
# HOW:  Test default configs, mathematical properties, and performance baselines
# ============================================================================

# Mathematical Security Regression Tests (28 tests)
add_executable(regression_security_test_math_regression
    test_security_math_regression.cpp
)
target_link_libraries(regression_security_test_math_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_security_test_math_regression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_math_regression
         COMMAND regression_security_test_math_regression)
set_tests_properties(regression_security_test_math_regression PROPERTIES
    LABELS "regression"
    TIMEOUT 120
)

# ============================================================================
# Security Integration Framework Regression Tests (Phase SC-4)
# ============================================================================
# WHAT: Regression tests for Universal Security Integration Framework
# WHY:  Ensure API stability, performance baselines, and thread safety
# HOW:  Test default configs, mathematical correctness, and concurrent access
# ============================================================================

# Security Integration Framework Regression Tests (25 tests)
add_executable(regression_security_test_framework
    test_security_integration_framework_regression.cpp
)
target_link_libraries(regression_security_test_framework
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_test_framework
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_framework
         COMMAND regression_security_test_framework)
set_tests_properties(regression_security_test_framework PROPERTIES
    LABELS "regression;security;sc4"
    TIMEOUT 180
)
