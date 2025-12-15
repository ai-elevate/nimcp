# ============================================================================
# Security Unit Tests - Blood-Brain Barrier (Phase IS-1)
# ============================================================================
# WHAT: Unit tests for Neural Immune System perimeter defense
# WHY:  Verify individual BBB module functionality and security guarantees
# HOW:  Test each module in isolation: input gate, code signing, memory, access
# ============================================================================

# Blood-Brain Barrier Unit Tests (79 tests)
add_executable(unit_security_test_blood_brain_barrier
    test_blood_brain_barrier.cpp
)
target_link_libraries(unit_security_test_blood_brain_barrier
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_blood_brain_barrier
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_blood_brain_barrier
         COMMAND unit_security_test_blood_brain_barrier)
set_tests_properties(unit_security_test_blood_brain_barrier PROPERTIES
    LABELS "unit"
    TIMEOUT 60
)

# BBB-Immune Integration Tests
add_executable(unit_security_test_bbb_immune_integration
    test_bbb_immune_integration.cpp
)
target_link_libraries(unit_security_test_bbb_immune_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_bbb_immune_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_bbb_immune_integration
         COMMAND unit_security_test_bbb_immune_integration)
set_tests_properties(unit_security_test_bbb_immune_integration PROPERTIES
    LABELS "unit;integration;security;immune"
    TIMEOUT 60
)

# General Security Unit Tests (legacy)
add_executable(unit_security_test_security
    test_security.cpp
)
target_link_libraries(unit_security_test_security
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_security
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security
         COMMAND unit_security_test_security)
set_tests_properties(unit_security_test_security PROPERTIES LABELS "unit")

# Comprehensive Security Unit Tests (legacy)
add_executable(unit_security_test_security_comprehensive
    test_security_comprehensive.cpp
)
target_link_libraries(unit_security_test_security_comprehensive
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_security_comprehensive
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security_comprehensive
         COMMAND unit_security_test_security_comprehensive)
set_tests_properties(unit_security_test_security_comprehensive PROPERTIES
    LABELS "unit"
    TIMEOUT 60
)

# Security Coverage Framework Unit Tests (Phase SC-1)
add_executable(unit_security_test_security_coverage
    test_security_coverage.cpp
)
target_link_libraries(unit_security_test_security_coverage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_security_coverage
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security_coverage
         COMMAND unit_security_test_security_coverage)
set_tests_properties(unit_security_test_security_coverage PROPERTIES
    LABELS "unit;security;sc1"
    TIMEOUT 120
)

# ============================================================================
# Security Integration Framework Unit Tests (Phase SC-4)
# ============================================================================
# WHAT: Unit tests for Universal Security Integration Framework
# WHY:  Verify module registration, trust management, DP queries, events
# HOW:  Test integration API for all module categories
# ============================================================================

add_executable(unit_security_test_security_integration
    test_security_integration.cpp
)
target_link_libraries(unit_security_test_security_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(unit_security_test_security_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_security_integration
         COMMAND unit_security_test_security_integration)
set_tests_properties(unit_security_test_security_integration PROPERTIES
    LABELS "unit;security;sc4"
    TIMEOUT 120
)

# ============================================================================
# Security State Management Unit Tests
# ============================================================================
# WHAT: Unit tests for Security Level Management
# WHY:  Verify downgrade prevention, level locking, component levels
# HOW:  Test security level API, feature enablement, audit trail
# ============================================================================

add_executable(unit_security_test_security_level
    test_security_level.cpp
)
target_link_libraries(unit_security_test_security_level
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(unit_security_test_security_level
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security_level
         COMMAND unit_security_test_security_level)
set_tests_properties(unit_security_test_security_level PROPERTIES
    LABELS "unit;security"
    TIMEOUT 60
)

# ============================================================================
# Distributed Security Consensus Unit Tests
# ============================================================================
# WHAT: Unit tests for Raft-like Security Consensus
# WHY:  Verify leader election, policy replication, threat sharing
# HOW:  Test consensus protocol, role transitions, cluster coordination
# ============================================================================

add_executable(unit_security_test_security_consensus
    test_security_consensus.cpp
)
target_link_libraries(unit_security_test_security_consensus
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(unit_security_test_security_consensus
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_security_test_security_consensus
         COMMAND unit_security_test_security_consensus)
set_tests_properties(unit_security_test_security_consensus PROPERTIES
    LABELS "unit;security"
    TIMEOUT 120
)

# ============================================================================
# Enhanced Security Framework Unit Tests (Phase SC-5)
# ============================================================================

# TOCTOU + Rate Limiting Unit Tests
# add_executable(unit_security_test_rate_limiter test_rate_limiter.cpp)
# target_link_libraries(unit_security_test_rate_limiter PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(unit_security_test_rate_limiter PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME unit_security_test_rate_limiter COMMAND unit_security_test_rate_limiter)
# set_tests_properties(unit_security_test_rate_limiter PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Constant-Time Crypto Unit Tests
add_executable(unit_security_test_constant_time test_constant_time.cpp)
target_link_libraries(unit_security_test_constant_time PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_constant_time PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_constant_time COMMAND unit_security_test_constant_time)
set_tests_properties(unit_security_test_constant_time PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Key Derivation Unit Tests
add_executable(unit_security_test_key_derivation test_key_derivation.cpp)
target_link_libraries(unit_security_test_key_derivation PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_key_derivation PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_key_derivation COMMAND unit_security_test_key_derivation)
set_tests_properties(unit_security_test_key_derivation PROPERTIES LABELS "unit;security" TIMEOUT 120)

# Path Traversal Detection Unit Tests
add_executable(unit_security_test_path_traversal test_path_traversal.cpp)
target_link_libraries(unit_security_test_path_traversal PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_path_traversal PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_path_traversal COMMAND unit_security_test_path_traversal)
set_tests_properties(unit_security_test_path_traversal PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Shell Command Injection Detection Unit Tests
add_executable(unit_security_test_shell_detector test_shell_detector.cpp)
target_link_libraries(unit_security_test_shell_detector PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_shell_detector PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_shell_detector COMMAND unit_security_test_shell_detector)
set_tests_properties(unit_security_test_shell_detector PROPERTIES LABELS "unit;security" TIMEOUT 60)

# ML-Based Anomaly Detection Unit Tests
add_executable(unit_security_test_anomaly_detector test_anomaly_detector.cpp)
target_link_libraries(unit_security_test_anomaly_detector PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_anomaly_detector PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_anomaly_detector COMMAND unit_security_test_anomaly_detector)
set_tests_properties(unit_security_test_anomaly_detector PROPERTIES LABELS "unit;security" TIMEOUT 120)

# Bayesian Network Unit Tests
add_executable(unit_security_test_bayesian_network test_bayesian_network.cpp)
target_link_libraries(unit_security_test_bayesian_network PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_bayesian_network PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_bayesian_network COMMAND unit_security_test_bayesian_network)
set_tests_properties(unit_security_test_bayesian_network PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Dynamic Pattern Database Unit Tests
add_executable(unit_security_test_pattern_db test_pattern_db.cpp)
target_link_libraries(unit_security_test_pattern_db PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_pattern_db PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_pattern_db COMMAND unit_security_test_pattern_db)
set_tests_properties(unit_security_test_pattern_db PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Encrypted Audit Buffer Unit Tests
add_executable(unit_security_test_encrypted_audit test_encrypted_audit.cpp)
target_link_libraries(unit_security_test_encrypted_audit PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_encrypted_audit PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_encrypted_audit COMMAND unit_security_test_encrypted_audit)
set_tests_properties(unit_security_test_encrypted_audit PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Policy Parser Unit Tests
add_executable(unit_security_test_policy_parser test_policy_parser.cpp)
target_link_libraries(unit_security_test_policy_parser PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_policy_parser PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_policy_parser COMMAND unit_security_test_policy_parser)
set_tests_properties(unit_security_test_policy_parser PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Policy Evaluator Unit Tests
# TEMPORARILY DISABLED - Policy compiler has unimplemented node types
# TODO: Implement policy compiler for all AST node types
# add_executable(unit_security_test_policy_evaluator test_policy_evaluator.cpp)
# target_link_libraries(unit_security_test_policy_evaluator PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(unit_security_test_policy_evaluator PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME unit_security_test_policy_evaluator COMMAND unit_security_test_policy_evaluator)
# set_tests_properties(unit_security_test_policy_evaluator PROPERTIES LABELS "unit;security" TIMEOUT 60)

# Post-Quantum Cryptography Unit Tests
add_executable(unit_security_test_post_quantum test_post_quantum.cpp)
target_link_libraries(unit_security_test_post_quantum PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_post_quantum PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_post_quantum COMMAND unit_security_test_post_quantum)
set_tests_properties(unit_security_test_post_quantum PROPERTIES LABELS "unit;security" TIMEOUT 120)

# Supply Chain Security Unit Tests
add_executable(unit_security_test_supply_chain test_supply_chain.cpp)
target_link_libraries(unit_security_test_supply_chain PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_supply_chain PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_supply_chain COMMAND unit_security_test_supply_chain)
set_tests_properties(unit_security_test_supply_chain PROPERTIES LABELS "unit;security" TIMEOUT 60)

# NaN/Inf Validation Unit Tests
add_executable(unit_security_test_nan_inf_validation unit_security_test_nan_inf_validation.cpp)
target_link_libraries(unit_security_test_nan_inf_validation PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_nan_inf_validation PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_nan_inf_validation COMMAND unit_security_test_nan_inf_validation)
set_tests_properties(unit_security_test_nan_inf_validation PROPERTIES LABELS "unit;security" TIMEOUT 60)

# ============================================================================
# Security FEP Bridge Tests
# ============================================================================
# WHAT: Unit tests for Security-FEP bidirectional integration bridges
# WHY:  Ensure threat detection integrates with FEP predictions
# HOW:  Test lifecycle, effects, and security-specific FEP modulations
# ============================================================================

add_executable(unit_security_test_fep_bridges test_security_fep_bridges.cpp)
target_link_libraries(unit_security_test_fep_bridges PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(unit_security_test_fep_bridges PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME unit_security_test_fep_bridges COMMAND unit_security_test_fep_bridges)
set_tests_properties(unit_security_test_fep_bridges PROPERTIES LABELS "unit;security;fep" TIMEOUT 60)

# Security Immune Bridge Tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/immune/CMakeLists.txt")
    add_subdirectory(immune)
endif()
