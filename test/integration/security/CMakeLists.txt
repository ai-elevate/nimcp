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

# ============================================================================
# BBB Bio-Router Integration Tests
# ============================================================================
# WHAT: Integration tests for BBB security validation through bio-router
# WHY:  Verify security checks are properly applied to all bio-async messages
# HOW:  Test BBB validation at message boundaries and cross-module security
# ============================================================================

# BBB Bio-Router Integration Tests (9 tests)
add_executable(integration_security_test_bbb_bio_router
    test_bbb_bio_router_integration.cpp
)
target_link_libraries(integration_security_test_bbb_bio_router
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(integration_security_test_bbb_bio_router
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/async
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME integration_security_test_bbb_bio_router
         COMMAND integration_security_test_bbb_bio_router)
set_tests_properties(integration_security_test_bbb_bio_router PROPERTIES
    LABELS "integration;security;bbb;bio-router"
    TIMEOUT 180
)

# ============================================================================
# Security State Management Integration Tests
# ============================================================================
# TEMPORARILY DISABLED - Needs API migration to bio_router
# TODO: Update test_security_level_integration.cpp to use bio_router API
# ============================================================================

# add_executable(integration_security_test_security_level
#     test_security_level_integration.cpp
# )
# target_link_libraries(integration_security_test_security_level
#     PRIVATE
#         GTest::GTest
#         GTest::Main
#         nimcp
#         pthread
# )
# target_include_directories(integration_security_test_security_level
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/include/security
#         ${CMAKE_SOURCE_DIR}/src/security
#         ${CMAKE_SOURCE_DIR}/src/utils/memory
#         ${CMAKE_SOURCE_DIR}/src/utils/logging
#         ${CMAKE_SOURCE_DIR}/test
#         ${CMAKE_SOURCE_DIR}/test/utils
# )
# add_test(NAME integration_security_test_security_level
#          COMMAND integration_security_test_security_level)
# set_tests_properties(integration_security_test_security_level PROPERTIES
#     LABELS "integration;security"
#     TIMEOUT 180
# )

# ============================================================================
# Distributed Security Consensus Integration Tests
# ============================================================================
# TEMPORARILY DISABLED - Needs API migration
# TODO: Update test_security_consensus_integration.cpp
# ============================================================================

# add_executable(integration_security_test_security_consensus
#     test_security_consensus_integration.cpp
# )
# target_link_libraries(integration_security_test_security_consensus
#     PRIVATE
#         GTest::GTest
#         GTest::Main
#         nimcp
#         pthread
# )
# target_include_directories(integration_security_test_security_consensus
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/include/security
#         ${CMAKE_SOURCE_DIR}/src/security
#         ${CMAKE_SOURCE_DIR}/src/utils/memory
#         ${CMAKE_SOURCE_DIR}/src/utils/logging
#         ${CMAKE_SOURCE_DIR}/test
#         ${CMAKE_SOURCE_DIR}/test/utils
# )
# add_test(NAME integration_security_test_security_consensus
#          COMMAND integration_security_test_security_consensus)
# set_tests_properties(integration_security_test_security_consensus PROPERTIES
#     LABELS "integration;security"
#     TIMEOUT 300
# )

# ============================================================================
# Enhanced Security Framework Integration Tests (Phase SC-5)
# ============================================================================

# Constant-Time Crypto Integration Tests
add_executable(integration_security_test_constant_time test_constant_time_integration.cpp)
target_link_libraries(integration_security_test_constant_time PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(integration_security_test_constant_time PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME integration_security_test_constant_time COMMAND integration_security_test_constant_time)
set_tests_properties(integration_security_test_constant_time PROPERTIES LABELS "integration;security" TIMEOUT 120)

# Key Derivation Integration Tests
add_executable(integration_security_test_key_derivation test_key_derivation_integration.cpp)
target_link_libraries(integration_security_test_key_derivation PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(integration_security_test_key_derivation PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME integration_security_test_key_derivation COMMAND integration_security_test_key_derivation)
set_tests_properties(integration_security_test_key_derivation PROPERTIES LABELS "integration;security" TIMEOUT 180)

# Path Traversal Integration Tests
# TEMPORARILY DISABLED - Needs bio_async API migration
# add_executable(integration_security_test_path_traversal test_path_traversal_integration.cpp)
# target_link_libraries(integration_security_test_path_traversal PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(integration_security_test_path_traversal PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME integration_security_test_path_traversal COMMAND integration_security_test_path_traversal)
# set_tests_properties(integration_security_test_path_traversal PROPERTIES LABELS "integration;security" TIMEOUT 120)

# Shell Detector Integration Tests
# TEMPORARILY DISABLED - Needs bio_async API migration
# add_executable(integration_security_test_shell_detector test_shell_detector_integration.cpp)
# target_link_libraries(integration_security_test_shell_detector PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(integration_security_test_shell_detector PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME integration_security_test_shell_detector COMMAND integration_security_test_shell_detector)
# set_tests_properties(integration_security_test_shell_detector PROPERTIES LABELS "integration;security" TIMEOUT 120)

# Anomaly Detector Integration Tests
add_executable(integration_security_test_anomaly_detector test_anomaly_detector_integration.cpp)
target_link_libraries(integration_security_test_anomaly_detector PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(integration_security_test_anomaly_detector PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME integration_security_test_anomaly_detector COMMAND integration_security_test_anomaly_detector)
set_tests_properties(integration_security_test_anomaly_detector PROPERTIES LABELS "integration;security" TIMEOUT 180)

# Pattern Database Integration Tests
add_executable(integration_security_test_pattern_db test_pattern_db_integration.cpp)
target_link_libraries(integration_security_test_pattern_db PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(integration_security_test_pattern_db PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME integration_security_test_pattern_db COMMAND integration_security_test_pattern_db)
set_tests_properties(integration_security_test_pattern_db PROPERTIES LABELS "integration;security" TIMEOUT 120)

# Encrypted Audit Integration Tests
add_executable(integration_security_test_encrypted_audit test_encrypted_audit_integration.cpp)
target_link_libraries(integration_security_test_encrypted_audit PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(integration_security_test_encrypted_audit PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME integration_security_test_encrypted_audit COMMAND integration_security_test_encrypted_audit)
set_tests_properties(integration_security_test_encrypted_audit PROPERTIES LABELS "integration;security" TIMEOUT 120)

# Policy Engine Integration Tests
# TEMPORARILY DISABLED - Needs bio_async API migration
# add_executable(integration_security_test_policy_engine test_policy_engine_integration.cpp)
# target_link_libraries(integration_security_test_policy_engine PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(integration_security_test_policy_engine PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME integration_security_test_policy_engine COMMAND integration_security_test_policy_engine)
# set_tests_properties(integration_security_test_policy_engine PROPERTIES LABELS "integration;security" TIMEOUT 180)

# Post-Quantum Integration Tests
# TEMPORARILY DISABLED - Needs include path fix
# add_executable(integration_security_test_post_quantum test_post_quantum_integration.cpp)
# target_link_libraries(integration_security_test_post_quantum PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(integration_security_test_post_quantum PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME integration_security_test_post_quantum COMMAND integration_security_test_post_quantum)
# set_tests_properties(integration_security_test_post_quantum PROPERTIES LABELS "integration;security" TIMEOUT 180)

# Supply Chain Integration Tests
# TEMPORARILY DISABLED - Needs API fixes
# add_executable(integration_security_test_supply_chain test_supply_chain_integration.cpp)
# target_link_libraries(integration_security_test_supply_chain PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(integration_security_test_supply_chain PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME integration_security_test_supply_chain COMMAND integration_security_test_supply_chain)
# set_tests_properties(integration_security_test_supply_chain PROPERTIES LABELS "integration;security" TIMEOUT 120)

# ============================================================================
# Swarm Security Integration Tests
# ============================================================================
# WHAT: Integration tests for BBB security in swarm communications
# WHY:  Verify end-to-end security across all swarm modules
# HOW:  Test complete attack scenarios and security workflows
# ============================================================================

# Swarm Security Integration Tests
add_executable(integration_security_test_swarm_security
    test_swarm_security_integration.cpp
)
target_link_libraries(integration_security_test_swarm_security
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(integration_security_test_swarm_security
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/swarm
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/swarm
        ${CMAKE_SOURCE_DIR}/src/async
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME integration_security_test_swarm_security
         COMMAND integration_security_test_swarm_security)
set_tests_properties(integration_security_test_swarm_security PROPERTIES
    LABELS "integration;security;swarm"
    TIMEOUT 300
)

# ============================================================================
# Security Sleep Bridge Integration Tests
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sleep/CMakeLists.txt")
    add_subdirectory(sleep)
endif()
