# ============================================================================
# Security Integration Tests - Blood-Brain Barrier (Phase IS-1)
# ============================================================================
# WHAT: Integration tests for Neural Immune System with NIMCP brain
# WHY:  Verify BBB integration with brain subsystems and cross-layer operation
# HOW:  Test multi-layer defense, brain coordination, and full system workflows
# ============================================================================

# Blood-Brain Barrier Integration Tests (16 tests)
add_executable(integration_security_test_bbb_integration
    test_bbb_integration.cpp
)
target_link_libraries(integration_security_test_bbb_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_security_test_bbb_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME integration_security_test_bbb_integration
         COMMAND integration_security_test_bbb_integration)
set_tests_properties(integration_security_test_bbb_integration PROPERTIES
    LABELS "integration"
    TIMEOUT 120
)
