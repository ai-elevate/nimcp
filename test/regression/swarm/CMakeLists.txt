# ============================================================================
# Swarm Regression Tests
# ============================================================================

# Test 1: Memory Regression
add_test_binary(
    test_swarm_memory_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_memory_regression.cpp
    regression
)

# Test 2: Latency Regression
add_test_binary(
    test_swarm_latency_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_latency_regression.cpp
    regression
)

# Test 3: Scalability Regression
add_test_binary(
    test_swarm_scalability_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_scalability_regression.cpp
    regression
)

# Test 4: Conflict Resolution Regression
add_test_binary(
    test_swarm_conflict_resolution_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_conflict_resolution_regression.cpp
    regression
)

# Test 5: Quorum and Immune Logic Regression
add_test_binary(
    test_quorum_immune_logic_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quorum_immune_logic_regression.cpp
    regression
)

# Test 6: Swarm Consciousness Regression
add_test_binary(
    test_swarm_consciousness_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_consciousness_regression.cpp
    regression
)

# Test 7: Enhanced Swarm Consciousness Regression (Peer Callbacks, Remote Phi, Dynamics)
add_test_binary(
    test_swarm_consciousness_enhanced_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_consciousness_enhanced_regression.cpp
    regression
)

# Test 8: Swarm Sleep and Immune Bridges Regression
add_executable(regression_swarm_bridges
    test_swarm_bridges_regression.cpp
)
target_link_libraries(regression_swarm_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_swarm_bridges
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/swarm/sleep
        ${CMAKE_SOURCE_DIR}/src/swarm/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/utils/thread
)
add_test(NAME regression_swarm_bridges
         COMMAND regression_swarm_bridges)
set_tests_properties(regression_swarm_bridges PROPERTIES LABELS "regression")

# Test: Enhanced Swarm Logic Bridge Regression
add_test_binary(
    test_swarm_logic_bridge_enhanced_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_logic_bridge_enhanced_regression.cpp
    regression
)

# Test: Swarm Brain Local Regression (Role-Based Memory Optimization)
add_test_binary(
    test_swarm_brain_local_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_brain_local_regression.cpp
    regression
)
