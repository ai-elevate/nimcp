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

# ============================================================================
# Mathematical Security Integration Tests (Phase SC-3)
# ============================================================================
# WHAT: Integration tests for Entropy, Bayesian Trust, and Differential Privacy
# WHY:  Verify cross-component workflows and concurrent access patterns
# HOW:  Test full security pipeline with all three mathematical mechanisms
# ============================================================================

# Mathematical Security Integration Tests (15 tests)
add_executable(integration_security_test_math_integration
    test_security_math_integration.cpp
)
target_link_libraries(integration_security_test_math_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(integration_security_test_math_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME integration_security_test_math_integration
         COMMAND integration_security_test_math_integration)
set_tests_properties(integration_security_test_math_integration PROPERTIES
    LABELS "integration"
    TIMEOUT 180
)

# ============================================================================
# Security Integration Framework Tests (Phase SC-4)
# ============================================================================
# WHAT: Integration tests for Universal Security Integration Framework
# WHY:  Verify cross-module security workflows and concurrent operations
# HOW:  Test multi-module registration, trust propagation, and full pipeline
# ============================================================================

# Security Integration Framework Integration Tests (13 tests)
add_executable(integration_security_test_framework
    test_security_integration_framework.cpp
)
target_link_libraries(integration_security_test_framework
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(integration_security_test_framework
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME integration_security_test_framework
         COMMAND integration_security_test_framework)
set_tests_properties(integration_security_test_framework PROPERTIES
    LABELS "integration;security;sc4"
    TIMEOUT 300
)
