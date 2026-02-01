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

# ============================================================================
# Security Bridges Regression Tests
# ============================================================================
# WHAT: Comprehensive regression tests for security bridges
# WHY:  Ensure backward compatibility and consistent behavior for:
#       - Security-Async Bridge (bio-async router integration)
#       - Security-Logging Bridge (audit trail)
#       - Security-Immune Unified Bridge (immune system integration)
# HOW:  Test API stability, configuration, detection, performance,
#       statistics, state machines, and bidirectional flow
# ============================================================================

add_executable(regression_security_bridges
    test_security_bridges_regression.cpp
)
target_link_libraries(regression_security_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_bridges
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/async
        ${CMAKE_SOURCE_DIR}/include/security/logging
        ${CMAKE_SOURCE_DIR}/include/security/immune
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/security/async
        ${CMAKE_SOURCE_DIR}/src/security/logging
        ${CMAKE_SOURCE_DIR}/src/security/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_bridges
         COMMAND regression_security_bridges)
set_tests_properties(regression_security_bridges PROPERTIES
    LABELS "regression;security;bridges"
    TIMEOUT 300
)

# ============================================================================
# Security Module Bridge Regression Tests (Phase SMB)
# ============================================================================
# WHAT: Regression tests for Security Module Bridges
# WHY:  Ensure API stability and consistent behavior across versions
# HOW:  Test backward compatibility, performance baselines, and thread safety
#       for perception, language, executive, memory, training, and rcog bridges
# ============================================================================

add_executable(regression_security_module_bridges
    test_security_module_regression.cpp
)
target_link_libraries(regression_security_module_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_module_bridges
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/perception
        ${CMAKE_SOURCE_DIR}/include/security/language
        ${CMAKE_SOURCE_DIR}/include/security/executive
        ${CMAKE_SOURCE_DIR}/include/security/memory
        ${CMAKE_SOURCE_DIR}/include/security/training
        ${CMAKE_SOURCE_DIR}/include/security/rcog
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_module_bridges
         COMMAND regression_security_module_bridges)
set_tests_properties(regression_security_module_bridges PROPERTIES
    LABELS "regression;security;module;perception;language;executive;memory;training;rcog"
    TIMEOUT 300
)

# ============================================================================
# Security Module Comprehensive Regression Tests
# ============================================================================
# WHAT: Comprehensive regression tests to ensure security bridges don't regress
# WHY:  Guarantee consistent performance, accuracy, and security properties
# HOW:  Performance benchmarks, detection rate validation, memory checks,
#       false positive monitoring, security guarantee verification, integration stability
#
# TEST CATEGORIES:
# 1. Performance Regression    - Timing bounds for security operations
# 2. Memory Regression         - No leaks in create/destroy cycles
# 3. Detection Accuracy        - Maintain detection rates >= baseline
# 4. False Positive Regression - Keep false positive rates low
# 5. Security Guarantees       - Critical security properties hold
# 6. Integration Stability     - Multi-bridge scenarios remain stable
# ============================================================================

add_executable(regression_security_module
    test_security_module_regression.cpp
)
target_link_libraries(regression_security_module
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_module
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/perception
        ${CMAKE_SOURCE_DIR}/include/security/language
        ${CMAKE_SOURCE_DIR}/include/security/executive
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_module
         COMMAND regression_security_module)
set_tests_properties(regression_security_module PROPERTIES
    LABELS "regression;security;performance;memory;accuracy;integration"
    TIMEOUT 600
)

# ============================================================================
# Security-Knowledge Graph Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Knowledge Graph Bridge
# WHY:  Ensure consistent performance, API stability, and security properties
# HOW:  Test query latency, detection rates, memory usage, API stability
# ============================================================================

add_executable(regression_security_knowledge_graph
    test_security_knowledge_graph_regression.cpp
)
target_link_libraries(regression_security_knowledge_graph
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_knowledge_graph
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/knowledge
        ${CMAKE_SOURCE_DIR}/include/cognitive/knowledge
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_knowledge_graph PRIVATE cxx_std_17)
add_test(NAME regression_security_knowledge_graph
         COMMAND regression_security_knowledge_graph)
set_tests_properties(regression_security_knowledge_graph PROPERTIES
    LABELS "regression;security;knowledge;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Collective Cognition Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Collective Cognition Bridge
# WHY:  Ensure consistent performance, API stability, and security properties
# HOW:  Test invariants, state consistency, boundary conditions, thread safety
# ============================================================================

add_executable(regression_security_collective
    test_security_collective_regression.cpp
)
target_link_libraries(regression_security_collective
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_collective
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/collective
        ${CMAKE_SOURCE_DIR}/include/cognitive/collective_cognition
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/include/utils
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_collective PRIVATE cxx_std_17)
add_test(NAME regression_security_collective
         COMMAND regression_security_collective)
set_tests_properties(regression_security_collective PROPERTIES
    LABELS "regression;security;collective;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Game Theory Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Game Theory Bridge
# WHY:  Ensure consistent performance, API stability, and security properties
# HOW:  Test performance thresholds, detection rates, memory usage, stats accuracy
# ============================================================================

add_executable(regression_security_game_theory
    test_security_game_theory_regression.cpp
)
target_link_libraries(regression_security_game_theory
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_game_theory
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/game_theory
        ${CMAKE_SOURCE_DIR}/include/cognitive/game_theory
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_game_theory PRIVATE cxx_std_17)
add_test(NAME regression_security_game_theory
         COMMAND regression_security_game_theory)
set_tests_properties(regression_security_game_theory PROPERTIES
    LABELS "regression;security;game_theory;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Imagination Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Imagination Bridge
# WHY:  Ensure consistent performance, API stability, and security properties
# HOW:  Test performance thresholds, detection rates, memory usage, stats accuracy
# ============================================================================

add_executable(regression_security_imagination
    test_security_imagination_regression.cpp
)
target_link_libraries(regression_security_imagination
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_imagination
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/imagination
        ${CMAKE_SOURCE_DIR}/include/cognitive/imagination
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_imagination PRIVATE cxx_std_17)
add_test(NAME regression_security_imagination
         COMMAND regression_security_imagination)
set_tests_properties(regression_security_imagination PROPERTIES
    LABELS "regression;security;imagination;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Epistemic Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Epistemic Bridge
# WHY:  Ensure API stability, performance baselines, and security guarantees
# HOW:  Test default configs, boundary conditions, performance baselines
# ============================================================================

add_executable(regression_security_epistemic
    test_security_epistemic_regression.cpp
)
target_link_libraries(regression_security_epistemic
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_epistemic
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/epistemic
        ${CMAKE_SOURCE_DIR}/include/cognitive/epistemic
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/include/utils/bridge
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_epistemic PRIVATE cxx_std_17)
add_test(NAME regression_security_epistemic
         COMMAND regression_security_epistemic)
set_tests_properties(regression_security_epistemic PROPERTIES
    LABELS "regression;security;epistemic;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Distributed Training Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Distributed Training Bridge
# WHY:  Ensure API stability, performance baselines, and security guarantees
# HOW:  Test Byzantine detection accuracy, worker trust consistency, gradient
#       validation bounds, checkpoint integrity, memory stability, and thread safety
# ============================================================================

add_executable(regression_security_distributed_training
    test_security_distributed_training_regression.cpp
)
target_link_libraries(regression_security_distributed_training
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_distributed_training
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/distributed
        ${CMAKE_SOURCE_DIR}/include/training
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_distributed_training PRIVATE cxx_std_17)
add_test(NAME regression_security_distributed_training
         COMMAND regression_security_distributed_training)
set_tests_properties(regression_security_distributed_training PROPERTIES
    LABELS "regression;security;distributed;training;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Continual Learning Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Continual Learning Bridge
# WHY:  Ensure API stability, performance baselines, and security guarantees
# HOW:  Test catastrophic forgetting protection accuracy, concept drift validation
#       consistency, memory replay integrity bounds, retention monitoring stability,
#       learning rate manipulation detection, memory stability, and thread safety
# ============================================================================

add_executable(regression_security_continual_learning
    test_security_continual_learning_regression.cpp
)
target_link_libraries(regression_security_continual_learning
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_continual_learning
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/continual
        ${CMAKE_SOURCE_DIR}/include/training
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/include/utils/bridge
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_continual_learning PRIVATE cxx_std_17)
add_test(NAME regression_security_continual_learning
         COMMAND regression_security_continual_learning)
set_tests_properties(regression_security_continual_learning PROPERTIES
    LABELS "regression;security;continual;learning;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Security-Hippocampus Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for Security-Hippocampus Bridge (Memory Consolidation Defense)
# WHY:  Ensure API stability, performance baselines, and security guarantees
# HOW:  Test memory leaks, thread safety, numerical stability, boundary conditions,
#       circular buffer handling, state corruption prevention, and performance stability
# ============================================================================

add_executable(regression_security_hippocampus
    test_security_hippocampus_regression.cpp
)
target_link_libraries(regression_security_hippocampus
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_hippocampus
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/security/hippocampus
        ${CMAKE_SOURCE_DIR}/include/core/hippocampus
        ${CMAKE_SOURCE_DIR}/include/cognitive/memory
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_hippocampus PRIVATE cxx_std_17)
add_test(NAME regression_security_hippocampus
         COMMAND regression_security_hippocampus)
set_tests_properties(regression_security_hippocampus PROPERTIES
    LABELS "regression;security;hippocampus;performance;memory"
    TIMEOUT 300
)

# ============================================================================
# Safety Enhancement Regression Tests (AI Safety Infrastructure)
# ============================================================================
# WHAT: Regression tests for 10 AI safety enhancement modules
# WHY:  Prevent regression of critical safety properties and performance
# HOW:  Test performance baselines, thread safety, safety invariants,
#       known vulnerability patterns, and edge cases from AI safety literature
# ============================================================================

add_executable(regression_security_safety_enhancement
    test_safety_enhancement_regression.cpp
)
target_link_libraries(regression_security_safety_enhancement
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)
target_include_directories(regression_security_safety_enhancement
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/async
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(regression_security_safety_enhancement PRIVATE cxx_std_17)
add_test(NAME regression_security_safety_enhancement
         COMMAND regression_security_safety_enhancement)
set_tests_properties(regression_security_safety_enhancement PROPERTIES
    LABELS "regression;security;safety;enhancement;performance"
    TIMEOUT 300
)
