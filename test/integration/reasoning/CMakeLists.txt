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

# =============================================================================
# NIMCP Reasoning Chain - Hypothalamus Bridge Integration Tests
# =============================================================================
# WHAT: Integration tests for the Hypo-Reasoning bridge with a live brain
# WHY:  Verify that modulation computation/application correctly adapts the
#       reasoning engine configuration when connected to a real brain
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(integration_reasoning_test_hypo_bridge
    test_reasoning_hypo_bridge_integration.cpp
)

target_link_libraries(integration_reasoning_test_hypo_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_reasoning_test_hypo_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_reasoning_test_hypo_bridge PRIVATE cxx_std_17)

gtest_discover_tests(integration_reasoning_test_hypo_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;reasoning;hypo-bridge"
        TIMEOUT 180
)

# =============================================================================
# NIMCP Reasoning Chain - Mesh Bridge Integration Tests
# =============================================================================
# WHAT: Integration tests for the Mesh-Reasoning bridge with a live brain
# WHY:  Verify that evidence gathering/application correctly adapts the
#       reasoning engine configuration when connected to a real brain
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(integration_reasoning_test_mesh_bridge
    test_reasoning_mesh_bridge_integration.cpp
)

target_link_libraries(integration_reasoning_test_mesh_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_reasoning_test_mesh_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_reasoning_test_mesh_bridge PRIVATE cxx_std_17)

gtest_discover_tests(integration_reasoning_test_mesh_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;reasoning;mesh-bridge"
        TIMEOUT 180
)

# =============================================================================
# NIMCP Reasoning Chain - Convergent Evidence Accumulation Integration Tests
# =============================================================================
# WHAT: Integration tests for convergent reasoning with a live brain
# WHY:  Verify multi-module contributor activation, fallback paths,
#       bridge interactions, and training API integration
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(integration_reasoning_test_convergent
    test_reasoning_convergent_integration.cpp
)

target_link_libraries(integration_reasoning_test_convergent
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_reasoning_test_convergent
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_reasoning_test_convergent PRIVATE cxx_std_17)

gtest_discover_tests(integration_reasoning_test_convergent
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;reasoning;convergent"
        TIMEOUT 180
)
