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

# Test 9: Swarm Conflict Resolution
add_test_binary(
    test_swarm_conflict_resolution
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_conflict_resolution.cpp
    unit
)

# Test 10: Quorum Logic Validation
add_test_binary(
    test_quorum_logic
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quorum_logic.cpp
    unit
)

# Test 11: Immune Threat Logic
add_test_binary(
    test_immune_threat_logic
    ${CMAKE_CURRENT_SOURCE_DIR}/test_immune_threat_logic.cpp
    unit
)

# Test 12: Swarm Consciousness (Gestalt)
add_test_binary(
    test_swarm_consciousness
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_consciousness.cpp
    unit
)

# Test 13: Enhanced Swarm Consciousness (Peer Callbacks, Remote Phi, Dynamics)
add_test_binary(
    test_swarm_consciousness_enhanced
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_consciousness_enhanced.cpp
    unit
)

# Test 14: Swarm Module Registry
add_test_binary(
    test_swarm_module_registry
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_module_registry.cpp
    unit
)

# ============================================================================
# FEP Bridge Tests
# ============================================================================

# Test FEP-1: Swarm Brain FEP Bridge
add_test_binary(
    test_swarm_brain_fep_bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_brain_fep_bridge.cpp
    unit
)

# Test FEP-2: Swarm Consciousness FEP Bridge
add_test_binary(
    test_swarm_consciousness_fep_bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_consciousness_fep_bridge.cpp
    unit
)

# Test FEP-3: All Swarm FEP Bridges (Combined)
add_test_binary(
    test_swarm_fep_bridges
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_fep_bridges.cpp
    unit
)

# Test: Enhanced Swarm Logic Bridge
add_test_binary(
    test_swarm_logic_bridge_enhanced
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_logic_bridge_enhanced.cpp
    unit
)

# Test 15: Quantum Consensus (Grover-inspired amplitude amplification)
add_test_binary(
    test_quantum_consensus
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quantum_consensus.cpp
    unit
)

# Test 16: Swarm Task Allocation System
add_test_binary(
    test_swarm_task
    ${CMAKE_CURRENT_SOURCE_DIR}/test_swarm_task.cpp
    unit
)

# ============================================================================
# Subdirectories
# ============================================================================

# Sleep Bridge Tests
add_subdirectory(sleep)

# Immune Bridge Tests
add_subdirectory(immune)
