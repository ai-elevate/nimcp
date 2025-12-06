# ============================================================================
# Reasoning Regression Tests
# ============================================================================

# Test 1: Reasoning performance
add_test_binary(
    test_reasoning_performance
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_performance.cpp
    regression
)
# Test 2: Reasoning memory
add_test_binary(
    test_reasoning_memory
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_memory.cpp
    regression
)
# Test 3: Rule learning accuracy
add_test_binary(
    test_rule_learning_accuracy
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rule_learning_accuracy.cpp
    regression
)
# Test 4: Proof finding speed
add_test_binary(
    test_proof_finding_speed
    ${CMAKE_CURRENT_SOURCE_DIR}/test_proof_finding_speed.cpp
    regression
)
# Test 5: Reasoning regression
add_test_binary(
    test_reasoning_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_regression.cpp
    regression
)
