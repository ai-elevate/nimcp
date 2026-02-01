# ============================================================================
# Mesh Network Integration Tests
# ============================================================================
# Hyperledger-inspired mesh network architecture integration tests
# Tests cross-component interactions and workflows
# ============================================================================

# Test 1: Mesh Channel Integration - World state, gossip, participants
add_test_binary(
    test_mesh_channel_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_channel_integration.cpp
    integration
)

# Test 2: Mesh Coordinator Pool Integration - BFT election, load balancing, failover
add_test_binary(
    test_mesh_coordinator_pool_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_coordinator_pool_integration.cpp
    integration
)

# Test 3: Mesh Transaction Flow Integration - Complete EOV lifecycle
add_test_binary(
    test_mesh_transaction_flow_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_transaction_flow_integration.cpp
    integration
)

# Test 4: Mesh MSP Integration - BBB, immune, credentials, quarantine
add_test_binary(
    test_mesh_msp_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_msp_integration.cpp
    integration
)

# ============================================================================
# Bootstrap Integration Tests
# ============================================================================

# Test 5: Mesh Bootstrap Integration - Full system bootstrap with all components
add_test_binary(
    test_mesh_bootstrap_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_bootstrap_integration.cpp
    integration
)

# ============================================================================
# Enhanced Pattern Routing Integration Tests
# ============================================================================

# Test 6: Enhanced Routing Integration - SAT solver + pattern cache together
add_test_binary(
    test_mesh_enhanced_routing_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_enhanced_routing_integration.cpp
    integration
)

# ============================================================================
# KG-Mesh Routing Bridge Integration Tests
# ============================================================================

# Test 7: KG Routing Bridge Integration - Hybrid routing with brain topology
add_test_binary(
    test_mesh_kg_routing_bridge_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_kg_routing_bridge_integration.cpp
    integration
)
