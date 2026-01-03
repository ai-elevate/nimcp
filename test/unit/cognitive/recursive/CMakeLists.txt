# ============================================================================
# Recursive Cognition Unit Tests
# ============================================================================
# WHAT: CMakeLists for recursive cognition (RLM-inspired) tests
# WHY:  Build GTest-based tests for context store and answer refiner
# HOW:  Add test binaries for each component
# ============================================================================

# Context Store Tests
add_test_binary(
    unit_cognitive_recursive_test_context_store
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rcog_context_store.cpp
    unit
)

# Answer Refiner Tests
add_test_binary(
    unit_cognitive_recursive_test_answer
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rcog_answer.cpp
    unit
)
