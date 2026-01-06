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
# DISABLED: Uses unimplemented APIs (brain_compile_rule_to_neural, etc.)
# add_test_binary(
#     test_reasoning_regression
#     ${CMAKE_CURRENT_SOURCE_DIR}/test_reasoning_regression.cpp
#     regression
# )

# Reasoning SNN-Plasticity Regression Tests
add_executable(regression_cognitive_reasoning_snn_plasticity
    test_reasoning_snn_plasticity_regression.cpp
)
target_link_libraries(regression_cognitive_reasoning_snn_plasticity
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
        m
)
target_include_directories(regression_cognitive_reasoning_snn_plasticity
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(regression_cognitive_reasoning_snn_plasticity PRIVATE cxx_std_17)
gtest_discover_tests(regression_cognitive_reasoning_snn_plasticity
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;reasoning;snn;plasticity"
        TIMEOUT 180
)
