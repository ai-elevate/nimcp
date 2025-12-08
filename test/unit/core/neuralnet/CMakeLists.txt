# ============================================================================
# Core Neuralnet Unit Tests
# ============================================================================

# Test 1: Sparse Synapse Allocation
add_test_binary(
    test_sparse_synapse
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_synapse.cpp
    unit
)
