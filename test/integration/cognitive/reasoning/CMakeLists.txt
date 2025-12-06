# ============================================================================
# Reasoning Integration Tests
# ============================================================================

# Test 1: End-to-end reasoning
add_test_binary(
    test_end_to_end_reasoning
    ${CMAKE_CURRENT_SOURCE_DIR}/test_end_to_end_reasoning.cpp
    integration
)
# Test 2: Neural-symbolic bridge
add_test_binary(
    test_neural_symbolic_bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/test_neural_symbolic_bridge.cpp
    integration
)
# Test 3: Brain reasoning API
add_test_binary(
    test_brain_reasoning_api
    ${CMAKE_CURRENT_SOURCE_DIR}/test_brain_reasoning_api.cpp
    integration
)
# Test 4: Cognitive event flow
add_test_binary(
    test_cognitive_event_flow
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cognitive_event_flow.cpp
    integration
)
# Test 5: Working memory reasoning
add_test_binary(
    test_working_memory_reasoning
    ${CMAKE_CURRENT_SOURCE_DIR}/test_working_memory_reasoning.cpp
    integration
)
# Test 6: Reasoning integration
add_test_binary(
    test_reasoning_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_integration.cpp
    integration
)
