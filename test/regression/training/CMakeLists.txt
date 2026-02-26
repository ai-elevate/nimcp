# =============================================================================
# NIMCP Training Integration Regression Tests
# =============================================================================
# WHAT: Regression tests guarding against training integration edge-case bugs
# WHY:  Prevent regressions in reward updates, adaptive LR, batch updates,
#       reasoning stability, and arousal bounds
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(regression_training_integration_test
    test_training_integration_regression.cpp
)

target_link_libraries(regression_training_integration_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_training_integration_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_training_integration_test PRIVATE cxx_std_17)

gtest_discover_tests(regression_training_integration_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;training;integration"
        TIMEOUT 300
)

# =============================================================================
# NIMCP Training Pipeline Regression Tests
# =============================================================================
# WHAT: Regression tests for the full training pipeline (public API + brain_ti_*)
# WHY:  Guards against label buffer overflow, gradient NaN, sparse synapse
#       segfaults, probe validity, brain state isolation, modulation NaN, and
#       post-batch stability.
# HOW:  GTest executable linked against nimcp library
#
# REGRESSIONS COVERED:
# 1. Label buffer overflow (nimcp_part_core.c:281)
# 2. Gradient distribution correctness
# 3. Sparse synapse learning no-segfault
# 4. Probe validity after learning
# 5. Multiple brains independence
# 6. Loss NaN/Inf detection
# 7. Zero-feature training safety
# 8. Label size boundary handling
# 9. Modulation state never NaN
# 10. Post-batch update stability
# =============================================================================

add_executable(regression_training_pipeline_test
    test_training_pipeline_regression.cpp
)

target_link_libraries(regression_training_pipeline_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_training_pipeline_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_training_pipeline_test PRIVATE cxx_std_17)

gtest_discover_tests(regression_training_pipeline_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;training;pipeline;modulation"
        TIMEOUT 300
        RESOURCE_LOCK "brain_heavy"
)
