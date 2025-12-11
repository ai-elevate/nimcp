# ============================================================================
# Core Neuralnet Unit Tests
# ============================================================================

# Test 1: Sparse Synapse Allocation
add_test_binary(
    test_sparse_synapse
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_synapse.cpp
    unit
)

# Test 2: Homeostasis-Immune Integration
add_test_binary(
    test_homeostasis_immune_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_homeostasis_immune_integration.cpp
    unit
)
