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

# =============================================================================
# NIMCP Reasoning Chain - Symbolic Logic Integration Tests
# =============================================================================
# WHAT: Integration tests for symbolic logic in the reasoning chain
# WHY:  Verify that symbolic query and inference phases actually query the
#       brain's formal knowledge base and produce SYMBOLIC_LOGIC steps
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(integration_reasoning_test_symbolic_logic
    test_reasoning_symbolic_logic_integration.cpp
)

target_link_libraries(integration_reasoning_test_symbolic_logic
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_reasoning_test_symbolic_logic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_reasoning_test_symbolic_logic PRIVATE cxx_std_17)

gtest_discover_tests(integration_reasoning_test_symbolic_logic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;reasoning;symbolic-logic"
        TIMEOUT 300
)

# =============================================================================
# NIMCP Reasoning Chain - Portia Bridge Integration Tests
# =============================================================================
# WHAT: Integration tests for the Portia-Reasoning bridge with a live brain
# WHY:  Verify that budget computation/application correctly adapts the
#       reasoning engine configuration when connected to a real brain
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(integration_reasoning_test_portia_bridge
    test_reasoning_portia_bridge_integration.cpp
)

target_link_libraries(integration_reasoning_test_portia_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_reasoning_test_portia_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_reasoning_test_portia_bridge PRIVATE cxx_std_17)

gtest_discover_tests(integration_reasoning_test_portia_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;reasoning;portia-bridge"
        TIMEOUT 180
)
