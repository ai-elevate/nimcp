# ============================================================================
# Reasoning Unit Tests
# ============================================================================

# Test 1: Symbolic Logic Brain Integration
add_test_binary(
    test_symbolic_logic_brain_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_symbolic_logic_brain_integration.cpp
    unit
)
# Test 2: Symbolic Logic Attachment
add_test_binary(
    test_symbolic_logic_attachment
    ${CMAKE_CURRENT_SOURCE_DIR}/test_symbolic_logic_attachment.cpp
    unit
)
# Test 3: Reasoning Factory
add_test_binary(
    test_reasoning_factory
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_factory.cpp
    unit
)
# Test 4: Forward Chaining
add_test_binary(
    test_forward_chaining
    ${CMAKE_CURRENT_SOURCE_DIR}/test_forward_chaining.cpp
    unit
)
# Test 5: Backward Chaining
add_test_binary(
    test_backward_chaining
    ${CMAKE_CURRENT_SOURCE_DIR}/test_backward_chaining.cpp
    unit
)
# Test 6: Unification Engine
add_test_binary(
    test_unification_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/test_unification_engine.cpp
    unit
)
# Test 7: Knowledge Base Interface
add_test_binary(
    test_knowledge_base_interface
    ${CMAKE_CURRENT_SOURCE_DIR}/test_knowledge_base_interface.cpp
    unit
)
# Test 8: Reasoning Unit Integration
add_test_binary(
    test_reasoning_unit_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_integration.cpp
    unit
)
# Test 9: Reasoning Attention Integration
add_test_binary(
    test_reasoning_attention
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_reasoning_attention.cpp
    unit
)
