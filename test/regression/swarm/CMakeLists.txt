# ============================================================================
# Swarm Regression Tests
# ============================================================================

# Test 1: Memory Regression
add_test_binary(
    test_swarm_memory_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_memory_regression.cpp
    regression
)

# Test 2: Latency Regression
add_test_binary(
    test_swarm_latency_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_latency_regression.cpp
    regression
)

# Test 3: Scalability Regression
add_test_binary(
    test_swarm_scalability_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_scalability_regression.cpp
    regression
)
