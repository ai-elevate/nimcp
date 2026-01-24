# ============================================================================
# Neuro-Symbolic Mathematics Integration Tests
# ============================================================================
# WHAT: Integration tests for Mathesis-inspired genius-level math system
# WHY:  Validate component interactions: energy consistency, hypergraph,
#       quantum MCTS, genius module, and orchestrator integration
# HOW:  Test data flow, cross-component operations, stress scenarios
# ============================================================================

# Main Integration Test
add_executable(integration_neuro_symbolic
    test_neuro_symbolic_integration.cpp
)

target_link_libraries(integration_neuro_symbolic
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_neuro_symbolic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(integration_neuro_symbolic PRIVATE cxx_std_17)

gtest_discover_tests(integration_neuro_symbolic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;neuro_symbolic"
        TIMEOUT 180
)

message(STATUS "  Added neuro-symbolic mathematics integration tests")
