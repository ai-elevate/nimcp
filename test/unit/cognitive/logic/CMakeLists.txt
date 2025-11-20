# ============================================================================
# Symbolic Logic Unit Tests
# ============================================================================

# Test 1: Symbolic Logic
add_test_binary(
    test_symbolic_logic
    ${CMAKE_CURRENT_SOURCE_DIR}/test_symbolic_logic.cpp
    unit
)

# Test 2: Symbolic Logic Real
add_test_binary(
    test_symbolic_logic_real
    ${CMAKE_CURRENT_SOURCE_DIR}/test_symbolic_logic_real.cpp
    unit
)

# Test 3: Neural Logic Real
add_test_binary(
    test_neural_logic_real
    ${CMAKE_CURRENT_SOURCE_DIR}/test_neural_logic_real.cpp
    unit
)
