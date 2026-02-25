# =============================================================================
# NIMCP Reasoning-Brain Integration Tests
# =============================================================================
# WHAT: Integration tests for reasoning engine connected to a live brain
# WHY:  Verify reasoning queries actually interact with brain state,
#       produce domain-aware conclusions, and accumulate statistics
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(integration_reasoning_test_reasoning_brain
    test_reasoning_brain_integration.cpp
)

target_link_libraries(integration_reasoning_test_reasoning_brain
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_reasoning_test_reasoning_brain
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_reasoning_test_reasoning_brain PRIVATE cxx_std_17)

gtest_discover_tests(integration_reasoning_test_reasoning_brain
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;reasoning;brain"
        TIMEOUT 180
)
