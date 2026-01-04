# ============================================================================
# Omnidirectional Inference Integration Tests
# ============================================================================
#
# WHAT: Integration tests for Phase 2 omnidirectional inference bridges
# WHY:  Verify all bridges (Sensory, CC, Occipital, Broca, KG) work together
# HOW:  Multi-bridge scenarios with JEPA, Hopfield, Hierarchy, and Replay

message(STATUS "Adding omnidirectional inference integration tests...")

# Integration Test: Multi-bridge omnidirectional inference
add_test_binary(
    test_omni_bridges_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_omni_bridges_integration.cpp
    integration
)

message(STATUS "  Configured omnidirectional inference integration tests")
