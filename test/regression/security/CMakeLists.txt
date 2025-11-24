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
