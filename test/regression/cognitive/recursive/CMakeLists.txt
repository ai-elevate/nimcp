# ============================================================================
# Recursive Cognition Regression Tests
# ============================================================================
#
# WHAT: Performance and correctness regression tests
# WHY:  Detect performance degradation and behavioral regressions
# HOW:  Benchmark operations, verify numerical stability

message(STATUS "Adding recursive cognition regression tests...")

# Regression Test: Performance and correctness
add_test_binary(
    test_rcog_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rcog_regression.cpp
    regression
)

# Set longer timeout for performance tests
set_tests_properties(test_rcog_regression PROPERTIES TIMEOUT 120)

message(STATUS "  Configured recursive cognition regression tests")
