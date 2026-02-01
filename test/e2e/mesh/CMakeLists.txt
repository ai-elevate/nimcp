# ============================================================================
# Mesh Network End-to-End Tests
# ============================================================================
# Full system integration tests for complete mesh network workflows
# Tests realistic multi-channel scenarios with all components
# ============================================================================

# Test 1: Full Transaction E2E - Complete EOV flow with all components
add_e2e_test(
    e2e_test_mesh_full_transaction
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_full_transaction_e2e.cpp
)

# Add to CTest with e2e label
add_test(NAME e2e_mesh_full_transaction COMMAND e2e_test_mesh_full_transaction)
set_tests_properties(e2e_mesh_full_transaction PROPERTIES LABELS "e2e;mesh")

# Test 2: MSP Security E2E - Credential lifecycle, BBB, immune integration
add_e2e_test(
    e2e_test_mesh_msp_security
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_msp_security_e2e.cpp
)

add_test(NAME e2e_mesh_msp_security COMMAND e2e_test_mesh_msp_security)
set_tests_properties(e2e_mesh_msp_security PROPERTIES LABELS "e2e;mesh;security")

# Test 3: System Integration E2E - Full brain hemisphere integration
add_e2e_test(
    e2e_test_mesh_system_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_system_integration_e2e.cpp
)

add_test(NAME e2e_mesh_system_integration COMMAND e2e_test_mesh_system_integration)
set_tests_properties(e2e_mesh_system_integration PROPERTIES LABELS "e2e;mesh;integration")

# ============================================================================
# Bootstrap E2E Tests
# ============================================================================

# Test 4: Bootstrap E2E - Full brain simulation scenarios
add_e2e_test(
    e2e_test_mesh_bootstrap
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_bootstrap_e2e.cpp
)

add_test(NAME e2e_mesh_bootstrap COMMAND e2e_test_mesh_bootstrap)
set_tests_properties(e2e_mesh_bootstrap PROPERTIES LABELS "e2e;mesh;bootstrap")

# ============================================================================
# Enhanced Pattern Routing E2E Tests
# ============================================================================

# Test 5: Enhanced Routing E2E - SAT solver + pattern cache full pipeline
add_e2e_test(
    e2e_test_mesh_enhanced_routing
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_enhanced_routing_e2e.cpp
)

add_test(NAME e2e_mesh_enhanced_routing COMMAND e2e_test_mesh_enhanced_routing)
set_tests_properties(e2e_mesh_enhanced_routing PROPERTIES LABELS "e2e;mesh;routing")
