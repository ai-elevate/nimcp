# ============================================================================
# Security Regression Tests - Blood-Brain Barrier (Phase IS-1)
# ============================================================================
# WHAT: Regression tests for Neural Immune System backward compatibility
# WHY:  Ensure BBB behavior remains consistent across versions
# HOW:  Test API stability, performance baselines, and security guarantees
# ============================================================================

# Blood-Brain Barrier Regression Tests (32 tests)
add_executable(regression_security_test_bbb_regression
    test_bbb_regression.cpp
)
target_link_libraries(regression_security_test_bbb_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_security_test_bbb_regression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_bbb_regression
         COMMAND regression_security_test_bbb_regression)
set_tests_properties(regression_security_test_bbb_regression PROPERTIES
    LABELS "regression"
    TIMEOUT 60
)

# ============================================================================
# Mathematical Security Regression Tests (Phase SC-3)
# ============================================================================
# WHAT: Regression tests for Entropy, Bayesian Trust, and Differential Privacy
# WHY:  Ensure mathematical correctness and API stability across versions
# HOW:  Test default configs, mathematical properties, and performance baselines
# ============================================================================

# Mathematical Security Regression Tests (28 tests)
add_executable(regression_security_test_math_regression
    test_security_math_regression.cpp
)
target_link_libraries(regression_security_test_math_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_security_test_math_regression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_math_regression
         COMMAND regression_security_test_math_regression)
set_tests_properties(regression_security_test_math_regression PROPERTIES
    LABELS "regression"
    TIMEOUT 120
)

# ============================================================================
# Security Integration Framework Regression Tests (Phase SC-4)
# ============================================================================
# WHAT: Regression tests for Universal Security Integration Framework
# WHY:  Ensure API stability, performance baselines, and thread safety
# HOW:  Test default configs, mathematical correctness, and concurrent access
# ============================================================================

# Security Integration Framework Regression Tests (25 tests)
add_executable(regression_security_test_framework
    test_security_integration_framework_regression.cpp
)
target_link_libraries(regression_security_test_framework
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_test_framework
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
add_test(NAME regression_security_test_framework
         COMMAND regression_security_test_framework)
set_tests_properties(regression_security_test_framework PROPERTIES
    LABELS "regression;security;sc4"
    TIMEOUT 180
)

# ============================================================================
# Security Validation Performance Regression Tests
# ============================================================================
# WHAT: Performance regression tests for BBB security validation
# WHY:  Ensure security features don't degrade system performance over time
# HOW:  Benchmark validation operations, measure memory usage, stress test
# ============================================================================

# Security Validation Performance Regression Tests (20 tests)
add_executable(regression_security_test_validation_performance
    test_security_validation_performance.cpp
)
target_link_libraries(regression_security_test_validation_performance
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_test_validation_performance
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_validation_performance
         COMMAND regression_security_test_validation_performance)
set_tests_properties(regression_security_test_validation_performance PROPERTIES
    LABELS "regression;security;performance"
    TIMEOUT 300
)

# ============================================================================
# Security State Management Regression Tests
# ============================================================================
# WHAT: Regression tests for Security Level Management
# WHY:  Ensure downgrade prevention, locking, and performance stability
# HOW:  Test memory leaks, thread safety, boundary conditions, performance
# ============================================================================

add_executable(regression_security_test_security_level
    test_security_level_regression.cpp
)
target_link_libraries(regression_security_test_security_level
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_test_security_level
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_security_level
         COMMAND regression_security_test_security_level)
set_tests_properties(regression_security_test_security_level PROPERTIES
    LABELS "regression;security"
    TIMEOUT 180
)

# ============================================================================
# Distributed Security Consensus Regression Tests
# ============================================================================
# WHAT: Regression tests for Multi-Node Security Consensus
# WHY:  Ensure consensus correctness, role transitions, and cluster stability
# HOW:  Test memory leaks, thread safety, election behavior, performance
# ============================================================================

add_executable(regression_security_test_security_consensus
    test_security_consensus_regression.cpp
)
target_link_libraries(regression_security_test_security_consensus
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_test_security_consensus
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_security_consensus
         COMMAND regression_security_test_security_consensus)
set_tests_properties(regression_security_test_security_consensus PROPERTIES
    LABELS "regression;security"
    TIMEOUT 240
)

# ============================================================================
# Enhanced Security Framework Regression Tests (Phase SC-5)
# ============================================================================

# Constant-Time Crypto Regression Tests
add_executable(regression_security_test_constant_time test_constant_time_regression.cpp)
target_link_libraries(regression_security_test_constant_time PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_constant_time PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_constant_time COMMAND regression_security_test_constant_time)
set_tests_properties(regression_security_test_constant_time PROPERTIES LABELS "regression;security" TIMEOUT 180)

# Key Derivation Regression Tests
add_executable(regression_security_test_key_derivation test_key_derivation_regression.cpp)
target_link_libraries(regression_security_test_key_derivation PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_key_derivation PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_key_derivation COMMAND regression_security_test_key_derivation)
set_tests_properties(regression_security_test_key_derivation PROPERTIES LABELS "regression;security" TIMEOUT 240)

# Path Traversal Regression Tests
add_executable(regression_security_test_path_traversal test_path_traversal_regression.cpp)
target_link_libraries(regression_security_test_path_traversal PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_path_traversal PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_path_traversal COMMAND regression_security_test_path_traversal)
set_tests_properties(regression_security_test_path_traversal PROPERTIES LABELS "regression;security" TIMEOUT 120)

# Shell Detector Regression Tests
add_executable(regression_security_test_shell_detector test_shell_detector_regression.cpp)
target_link_libraries(regression_security_test_shell_detector PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_shell_detector PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_shell_detector COMMAND regression_security_test_shell_detector)
set_tests_properties(regression_security_test_shell_detector PROPERTIES LABELS "regression;security" TIMEOUT 120)

# Anomaly Detector Regression Tests
add_executable(regression_security_test_anomaly_detector test_anomaly_detector_regression.cpp)
target_link_libraries(regression_security_test_anomaly_detector PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_anomaly_detector PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_anomaly_detector COMMAND regression_security_test_anomaly_detector)
set_tests_properties(regression_security_test_anomaly_detector PROPERTIES LABELS "regression;security" TIMEOUT 180)

# Pattern Database Regression Tests
add_executable(regression_security_test_pattern_db test_pattern_db_regression.cpp)
target_link_libraries(regression_security_test_pattern_db PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_pattern_db PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_pattern_db COMMAND regression_security_test_pattern_db)
set_tests_properties(regression_security_test_pattern_db PROPERTIES LABELS "regression;security" TIMEOUT 120)

# Encrypted Audit Regression Tests
add_executable(regression_security_test_encrypted_audit test_encrypted_audit_regression.cpp)
target_link_libraries(regression_security_test_encrypted_audit PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_encrypted_audit PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_encrypted_audit COMMAND regression_security_test_encrypted_audit)
set_tests_properties(regression_security_test_encrypted_audit PROPERTIES LABELS "regression;security" TIMEOUT 120)

# Policy Engine Regression Tests
# TEMPORARILY DISABLED - Policy compiler has unimplemented node types
# TODO: Implement policy compiler for all AST node types
# add_executable(regression_security_test_policy_engine test_policy_engine_regression.cpp)
# target_link_libraries(regression_security_test_policy_engine PRIVATE GTest::GTest GTest::Main nimcp pthread)
# target_include_directories(regression_security_test_policy_engine PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# add_test(NAME regression_security_test_policy_engine COMMAND regression_security_test_policy_engine)
# set_tests_properties(regression_security_test_policy_engine PROPERTIES LABELS "regression;security" TIMEOUT 180)

# Post-Quantum Regression Tests
add_executable(regression_security_test_post_quantum test_post_quantum_regression.cpp)
target_link_libraries(regression_security_test_post_quantum PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_post_quantum PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_post_quantum COMMAND regression_security_test_post_quantum)
set_tests_properties(regression_security_test_post_quantum PROPERTIES LABELS "regression;security" TIMEOUT 180)

# Supply Chain Regression Tests
add_executable(regression_security_test_supply_chain test_supply_chain_regression.cpp)
target_link_libraries(regression_security_test_supply_chain PRIVATE GTest::GTest GTest::Main nimcp pthread)
target_include_directories(regression_security_test_supply_chain PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME regression_security_test_supply_chain COMMAND regression_security_test_supply_chain)
set_tests_properties(regression_security_test_supply_chain PROPERTIES LABELS "regression;security" TIMEOUT 120)

# ============================================================================
# Security Sleep Bridge Regression Tests
# ============================================================================
add_executable(regression_security_sleep_bridges
    test_security_sleep_regression.cpp
)
target_link_libraries(regression_security_sleep_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_security_sleep_bridges
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security/sleep
        ${CMAKE_SOURCE_DIR}/src/cognitive
        ${CMAKE_SOURCE_DIR}/src/utils/thread
)
add_test(NAME regression_security_sleep_bridges
         COMMAND regression_security_sleep_bridges)
set_tests_properties(regression_security_sleep_bridges PROPERTIES LABELS "regression;security")
