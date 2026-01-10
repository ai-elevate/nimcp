# Hypothalamus Region Regression Tests
# Tests for alignment behavior consistency

# Helper macro for hypothalamus regression tests
macro(add_hypothalamus_regression_test TEST_NAME)
    add_executable(regression_core_brain_regions_hypothalamus_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(regression_core_brain_regions_hypothalamus_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
            pthread
    )
    add_test(NAME regression_core_brain_regions_hypothalamus_${TEST_NAME}
             COMMAND regression_core_brain_regions_hypothalamus_${TEST_NAME})
endmacro()

# Alignment regression tests (Phase 19)
add_hypothalamus_regression_test(test_hypothalamus_alignment_regression)

# Orchestrator regression tests
add_hypothalamus_regression_test(test_hypothalamus_orchestrator_regression)

# FEP bridges regression tests
add_hypothalamus_regression_test(test_hypo_fep_bridges_regression)

# Hypothalamus-Training bridge regression tests
add_hypothalamus_regression_test(test_hypothalamus_training_regression)

# Hypothalamus-Perception bridge regression tests (Phase 17.5)
add_hypothalamus_regression_test(test_hypothalamus_perception_bridge_regression)
