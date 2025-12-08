# ============================================================================
# Portia Spider System Regression Tests
# ============================================================================
# WHAT: Regression tests for Portia adaptive intelligence system
# WHY:  Ensure stability, performance, and correctness across versions
# HOW:  Comprehensive stress tests, fairness checks, accuracy verification
# ============================================================================

# Tier Switch Stability Tests
add_executable(regression_portia_test_tier_switch_stability
    test_portia_tier_switch_stability.cpp
)
target_link_libraries(regression_portia_test_tier_switch_stability
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_tier_switch_stability
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_tier_switch_stability
         COMMAND regression_portia_test_tier_switch_stability)
set_tests_properties(regression_portia_test_tier_switch_stability PROPERTIES
    LABELS "regression;portia"
    TIMEOUT 180
)

# Power Stability Tests
add_executable(regression_portia_test_power_stability
    test_portia_power_stability.cpp
)
target_link_libraries(regression_portia_test_power_stability
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_power_stability
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_power_stability
         COMMAND regression_portia_test_power_stability)
set_tests_properties(regression_portia_test_power_stability PROPERTIES
    LABELS "regression;portia"
    TIMEOUT 180
)

# Planning Performance Tests
add_executable(regression_portia_test_planning_performance
    test_portia_planning_performance.cpp
)
target_link_libraries(regression_portia_test_planning_performance
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_planning_performance
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_planning_performance
         COMMAND regression_portia_test_planning_performance)
set_tests_properties(regression_portia_test_planning_performance PROPERTIES
    LABELS "regression;portia;performance"
    TIMEOUT 240
)

# Attention Fairness Tests
add_executable(regression_portia_test_attention_fairness
    test_portia_attention_fairness.cpp
)
target_link_libraries(regression_portia_test_attention_fairness
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_attention_fairness
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_attention_fairness
         COMMAND regression_portia_test_attention_fairness)
set_tests_properties(regression_portia_test_attention_fairness PROPERTIES
    LABELS "regression;portia;fairness"
    TIMEOUT 240
)

# Accelerator Robustness Tests
add_executable(regression_portia_test_accelerator_robustness
    test_portia_accelerator_robustness.cpp
)
target_link_libraries(regression_portia_test_accelerator_robustness
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_accelerator_robustness
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_accelerator_robustness
         COMMAND regression_portia_test_accelerator_robustness)
set_tests_properties(regression_portia_test_accelerator_robustness PROPERTIES
    LABELS "regression;portia"
    TIMEOUT 120
)

# Learning Capacity Tests
add_executable(regression_portia_test_learning_capacity
    test_portia_learning_capacity.cpp
)
target_link_libraries(regression_portia_test_learning_capacity
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_learning_capacity
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_learning_capacity
         COMMAND regression_portia_test_learning_capacity)
set_tests_properties(regression_portia_test_learning_capacity PROPERTIES
    LABELS "regression;portia;capacity"
    TIMEOUT 180
)

# Sensor Fusion Accuracy Tests
add_executable(regression_portia_test_fusion_accuracy
    test_portia_fusion_accuracy.cpp
)
target_link_libraries(regression_portia_test_fusion_accuracy
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_fusion_accuracy
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_fusion_accuracy
         COMMAND regression_portia_test_fusion_accuracy)
set_tests_properties(regression_portia_test_fusion_accuracy PROPERTIES
    LABELS "regression;portia;accuracy"
    TIMEOUT 180
)

# Degradation Recovery Tests
add_executable(regression_portia_test_degradation_recovery
    test_portia_degradation_recovery.cpp
)
target_link_libraries(regression_portia_test_degradation_recovery
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_degradation_recovery
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_degradation_recovery
         COMMAND regression_portia_test_degradation_recovery)
set_tests_properties(regression_portia_test_degradation_recovery PROPERTIES
    LABELS "regression;portia"
    TIMEOUT 120
)

# Classification Accuracy Tests
add_executable(regression_portia_test_classification_accuracy
    test_portia_classification_accuracy.cpp
)
target_link_libraries(regression_portia_test_classification_accuracy
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_classification_accuracy
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_classification_accuracy
         COMMAND regression_portia_test_classification_accuracy)
set_tests_properties(regression_portia_test_classification_accuracy PROPERTIES
    LABELS "regression;portia;accuracy"
    TIMEOUT 120
)

# Deception Effectiveness Tests
add_executable(regression_portia_test_deception_effectiveness
    test_portia_deception_effectiveness.cpp
)
target_link_libraries(regression_portia_test_deception_effectiveness
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_portia_test_deception_effectiveness
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/portia
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME regression_portia_test_deception_effectiveness
         COMMAND regression_portia_test_deception_effectiveness)
set_tests_properties(regression_portia_test_deception_effectiveness PROPERTIES
    LABELS "regression;portia"
    TIMEOUT 60
)

# ============================================================================
# Test Summary
# ============================================================================
# Total Tests: 10 test executables
# Total Test Cases: ~100+ individual test cases
# Coverage Areas:
#   - Tier switching stability and performance
#   - Power monitoring accuracy and stability
#   - Planning performance and memory bounds
#   - Resource allocation fairness
#   - Hardware accelerator robustness
#   - Learning system capacity handling
#   - Sensor fusion accuracy
#   - Graceful degradation and recovery
#   - Target classification accuracy
#   - Deception system effectiveness
# ============================================================================
