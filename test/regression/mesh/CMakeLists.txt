# ============================================================================
# Mesh Network Regression Tests
# ============================================================================
# Tests for stability issues, race conditions, edge cases, and backward compat
# Catch regressions in mesh network behavior
# ============================================================================

# Test 1: Mesh Stability Regression - Race conditions, resource exhaustion, edge cases
add_test_binary(
    test_mesh_stability_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_stability_regression.cpp
    regression
)

# Test 2: Mesh Consensus Regression - BFT edge cases, split-brain, partition recovery
add_test_binary(
    test_mesh_consensus_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_consensus_regression.cpp
    regression
)

# Test 3: Mesh Transaction Regression - EOV edge cases, timeout handling, retry logic
add_test_binary(
    test_mesh_transaction_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_transaction_regression.cpp
    regression
)

# Test 4: Mesh Performance Regression - Throughput, latency, memory usage benchmarks
add_test_binary(
    test_mesh_performance_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_performance_regression.cpp
    regression
)

# ============================================================================
# Bootstrap Regression Tests
# ============================================================================

# Test 5: Mesh Bootstrap Regression - Stability, edge cases, race conditions
add_test_binary(
    test_mesh_bootstrap_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_bootstrap_regression.cpp
    regression
)

# ============================================================================
# Enhanced Pattern Routing Regression Tests
# ============================================================================

# Test 6: Enhanced Routing Regression - SAT solver + pattern cache stability
add_test_binary(
    test_mesh_enhanced_routing_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_enhanced_routing_regression.cpp
    regression
)
