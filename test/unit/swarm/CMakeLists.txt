# ============================================================================
# Swarm Unit Tests
# ============================================================================

# Test 1: Swarm Protocol
add_test_binary(
    test_swarm_protocol
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_protocol.cpp
    unit
)

# Test 2: Swarm Signal Adapter
add_test_binary(
    test_swarm_signal
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_signal.cpp
    unit
)

# Test 3: Collective Workspace
add_test_binary(
    test_collective_workspace
    ${CMAKE_CURRENT_SOURCE_DIR}/test_collective_workspace.cpp
    unit
)

# Test 4: Swarm Emergence
add_test_binary(
    test_swarm_emergence
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_emergence.cpp
    unit
)

# Test 5: Swarm Consensus
add_test_binary(
    test_swarm_consensus
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_consensus.cpp
    unit
)

# Test 6: Swarm Brain Coordinator
add_test_binary(
    test_swarm_brain
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_brain.cpp
    unit
)

# Test 7: Swarm Gateway
add_test_binary(
    test_swarm_gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_gateway.cpp
    unit
)

# Test 8: Swarm Security (BBB Integration)
add_test_binary(
    test_swarm_security
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_security.cpp
    unit
)
