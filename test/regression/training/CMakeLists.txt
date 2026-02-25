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
