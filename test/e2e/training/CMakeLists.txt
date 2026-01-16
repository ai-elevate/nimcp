# ============================================================================
# NIMCP Training E2E Tests
# ============================================================================
# WHAT: Build configuration for training validation end-to-end tests
# WHY:  Validate complete training infrastructure before production training
# HOW:  Test data loading, gradient flow, memory stability, convergence
#
# E2E TESTS:
# - e2e_test_training_validation: Comprehensive training pipeline validation
# ============================================================================

message(STATUS "  Adding Training Validation E2E tests...")

# ============================================================================
# E2E Test: Training Pipeline Validation
# ============================================================================
# WHAT: Comprehensive training validation test
# WHY:  Ensure all training components work correctly end-to-end
# HOW:  Tests data loading, gradient flow, weight updates, convergence
#
# VALIDATION POINTS:
# 1. End-to-end training loop functionality
# 2. Data loading pipeline correctness
# 3. Gradient flow and weight updates
# 4. Memory stability over extended training runs
# 5. Convergence on simple tasks (XOR, linear separation)
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_training_validation.cpp)
    add_executable(e2e_test_training_validation
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_training_validation.cpp
    )
    target_include_directories(e2e_test_training_validation PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/e2e
        ${CMAKE_SOURCE_DIR}/src/lib
        ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(e2e_test_training_validation PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_training_validation PRIVATE cxx_std_20)
    target_compile_definitions(e2e_test_training_validation PRIVATE
        NIMCP_TESTING
        NIMCP_E2E_TESTING
    )
    add_test(NAME e2e_test_training_validation COMMAND e2e_test_training_validation)
    set_tests_properties(e2e_test_training_validation PROPERTIES
        LABELS "e2e;training;validation;convergence"
        TIMEOUT 600
    )
    message(STATUS "    Added: e2e_test_training_validation")
endif()

# ============================================================================
# E2E Test: Training Dispatch Workflows (C)
# ============================================================================
# WHAT: End-to-end tests for training dispatch complete workflows
# WHY:  Verify training dispatch integrates correctly with all network types
# HOW:  Tests classification, network type switching, save/load, result structs
#
# TESTS:
# 1. test_e2e_classification_adaptive - Full ADAPTIVE training workflow
# 2. test_e2e_classification_with_network_type_switch - Switch network types
# 3. test_e2e_training_save_load - Save/load brain and continue training
# 4. test_e2e_training_result_struct - Verify training result population
# 5. test_e2e_full_training_pipeline - Complete pipeline with stats
# 6. test_e2e_network_type_via_public_api - Public API only verification
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_training_dispatch_e2e.c)
    add_executable(e2e_test_training_dispatch
        ${CMAKE_CURRENT_SOURCE_DIR}/test_training_dispatch_e2e.c
    )
    target_include_directories(e2e_test_training_dispatch PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/e2e
        ${CMAKE_SOURCE_DIR}/src/lib
    )
    target_link_libraries(e2e_test_training_dispatch PRIVATE
        nimcp
        Threads::Threads
        m
    )
    target_compile_definitions(e2e_test_training_dispatch PRIVATE
        NIMCP_TESTING
        NIMCP_E2E_TESTING
    )
    add_test(NAME e2e_test_training_dispatch COMMAND e2e_test_training_dispatch)
    set_tests_properties(e2e_test_training_dispatch PROPERTIES
        LABELS "e2e;training;dispatch;workflow"
        TIMEOUT 600
    )
    message(STATUS "    Added: e2e_test_training_dispatch")
endif()
