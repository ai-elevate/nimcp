# ============================================================================
# Recursive Cognition Unit Tests
# ============================================================================
#
# WHAT: Unit tests for recursive cognition module components
# WHY:  Verify individual bridge functionality and API contracts
# HOW:  GTest-based tests with comprehensive coverage

message(STATUS "Adding recursive cognition unit tests...")

# Test 1: All RCOG Bridges
add_test_binary(
    test_rcog_bridges
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rcog_bridges.cpp
    unit
)

message(STATUS "  Configured recursive cognition unit tests")
