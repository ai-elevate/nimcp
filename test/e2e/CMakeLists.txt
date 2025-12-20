# ============================================================================
# NIMCP End-to-End (E2E) Test Framework
# ============================================================================
# WHAT: Build configuration for end-to-end integration tests
# WHY:  Test complete pipelines across multiple subsystems
# HOW:  Link full NIMCP library with GTest for comprehensive testing
#
# E2E TESTS:
# - test_brain_pipeline: Brain lifecycle (create, train, infer, save, restore)
# - test_cognitive_pipeline: Cognitive processing flow (perception → decision → memory)
# - test_distributed_pipeline: Distributed brain synchronization
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping E2E tests")
    return()
endif()

message(STATUS "")
message(STATUS "===============================================================================")
message(STATUS "NIMCP E2E Test Framework")
message(STATUS "===============================================================================")

# ============================================================================
# Common E2E Test Configuration
# ============================================================================

set(E2E_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/test/e2e
    ${CMAKE_SOURCE_DIR}/src/lib
    ${GTEST_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

set(E2E_LIBRARIES
    nimcp_portia
    nimcp
    nimcp_middleware
    GTest::GTest
    GTest::Main
    Python3::Python
    pthread
    m
)

# ============================================================================
# Helper: Create E2E Test Binary
# ============================================================================
# WHAT: Create E2E test executable with full NIMCP library linkage
# WHY:  E2E tests require complete system integration
# HOW:  Link against full nimcp library with all dependencies
#
# PARAMETERS:
#   TARGET_NAME - Test executable name
#   SOURCE_FILES - List of test source files
# ============================================================================
function(add_e2e_test TARGET_NAME)
    set(SOURCE_FILES ${ARGN})

    # Guard: Validate sources exist
    foreach(SOURCE_FILE ${SOURCE_FILES})
        if(NOT EXISTS ${SOURCE_FILE})
            message(WARNING "E2E test source not found: ${SOURCE_FILE}")
            return()
        endif()
    endforeach()

    # Create executable
    add_executable(${TARGET_NAME} ${SOURCE_FILES})

    # Add explicit dependency on nimcp library
    add_dependencies(${TARGET_NAME} nimcp)

    # Configure includes
    target_include_directories(${TARGET_NAME} PRIVATE ${E2E_INCLUDE_DIRS})

    # Enable C++20 for designated initializers
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

    # Enable test-only APIs and E2E mode
    target_compile_definitions(${TARGET_NAME} PRIVATE
        NIMCP_TESTING
        NIMCP_E2E_TESTING
    )

    # Link libraries
    target_link_libraries(${TARGET_NAME} PRIVATE ${E2E_LIBRARIES})

    # Set properties
    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # Enable coverage for Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(CMAKE_COMPILER_IS_GNUCXX)
            target_compile_options(${TARGET_NAME} PRIVATE --coverage)
            target_link_options(${TARGET_NAME} PRIVATE --coverage)
        endif()
    endif()

    # Register with CTest (E2E tests run sequentially with longer timeout)
    add_test(NAME ${TARGET_NAME}
             COMMAND ${TARGET_NAME}
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Set E2E test properties (longer timeout, sequential execution)
    set_tests_properties(${TARGET_NAME} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python;LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp"
        LABELS "e2e"
        TIMEOUT 120  # 2 minutes for E2E tests
        RUN_SERIAL TRUE  # Run sequentially to avoid resource contention
    )

    message(STATUS "  Added E2E test: ${TARGET_NAME}")
endfunction()

# ============================================================================
# E2E Test Targets
# ============================================================================

# Build the E2E test framework support files
add_e2e_test(e2e_test_brain_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_brain_pipeline.cpp
)

add_e2e_test(e2e_test_cognitive_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cognitive_pipeline.cpp
)

add_e2e_test(e2e_test_distributed_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_pipeline.cpp
)

add_e2e_test(e2e_test_bio_async_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bio_async_pipeline.cpp
)

add_e2e_test(e2e_test_cognitive_bio_async
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cognitive_bio_async_e2e.cpp
)

add_e2e_test(e2e_test_core_bio_async
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_core_bio_async_e2e.cpp
)

add_e2e_test(e2e_test_bio_async_training_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_bio_async_training_pipeline.cpp
)

# Fixed: Lambda capture issues - now using static handlers with user_data
add_e2e_test(e2e_test_bio_async_plasticity_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_bio_async_plasticity_pipeline.cpp
)

# Fixed: Missing header path - now using cognitive/introspection/nimcp_introspection.h
add_e2e_test(e2e_test_bio_async_cognitive_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_bio_async_cognitive_pipeline.cpp
)

# Fixed: BBB functions are in nimcp library - should link correctly
add_e2e_test(e2e_test_security_attack_simulation
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_security_attack_simulation.cpp
)

# Swarm E2E Tests
add_e2e_test(e2e_test_swarm_formation
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_swarm_formation.cpp
)

add_e2e_test(e2e_test_swarm_threat_response
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_swarm_threat_response.cpp
)

add_e2e_test(e2e_test_server_swarm_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_server_swarm_pipeline.cpp
)

# Neural Link Protocol (NLP) E2E Tests
add_e2e_test(e2e_test_nlp_swarm_coordination
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_nlp_swarm_coordination.cpp
)

add_e2e_test(e2e_test_nlp_war_zone
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_nlp_war_zone.cpp
)

add_e2e_test(e2e_test_nlp_disaster_sar
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_nlp_disaster_sar.cpp
)

add_e2e_test(e2e_test_nlp_neural_distributed
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_nlp_neural_distributed.cpp
)

# Portia Spider E2E Tests
add_e2e_test(e2e_test_portia_constrained_platform
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_constrained_platform.cpp
)

add_e2e_test(e2e_test_portia_tier_lifecycle
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_tier_lifecycle.cpp
)

add_e2e_test(e2e_test_portia_power_lifecycle
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_power_lifecycle.cpp
)

add_e2e_test(e2e_test_portia_learning_adaptation
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_learning_adaptation.cpp
)

add_e2e_test(e2e_test_portia_sensor_fusion_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_sensor_fusion_pipeline.cpp
)

add_e2e_test(e2e_test_portia_degradation_scenario
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_degradation_scenario.cpp
)

add_e2e_test(e2e_test_portia_bio_async_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_bio_async_pipeline.cpp
)

add_e2e_test(e2e_test_portia_security_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_security_integration.cpp
)

add_e2e_test(e2e_test_portia_planning_mission
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_planning_mission.cpp
)

add_e2e_test(e2e_test_portia_threat_response
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_threat_response.cpp
)

add_e2e_test(e2e_test_portia_swarm_logic_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_portia_swarm_logic_pipeline.cpp
)

# Emotion Tensor E2E Test
add_e2e_test(e2e_test_emotion_tensor_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_emotion_tensor_pipeline.cpp
)

# Second Messenger Cascade E2E Test
add_e2e_test(e2e_test_second_messenger_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_second_messenger_pipeline.cpp
)

# Positional Encoding Pipeline E2E Test
add_e2e_test(e2e_test_positional_encoding_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_positional_encoding_pipeline.cpp
)

# Introspection Pipeline E2E Test (Consciousness, Temporal Patterns, Uncertainty)
add_e2e_test(e2e_test_introspection_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_introspection_pipeline.cpp
)

# Swarm Consciousness Pipeline E2E Test (Gestalt Consciousness)
add_e2e_test(e2e_test_swarm_consciousness_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_swarm_consciousness_pipeline.cpp
)

# Enhanced Swarm Consciousness Pipeline E2E Test (Peer Callbacks, Remote Phi, Dynamics)
add_e2e_test(e2e_test_swarm_consciousness_enhanced_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_swarm_consciousness_enhanced_pipeline.cpp
)

# Enhanced Fault Tolerance Pipeline E2E Test (Immune System Integration)
add_e2e_test(e2e_test_fault_tolerance_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_fault_tolerance_pipeline.cpp
)

# Brain Immune System Pipeline E2E Test (B cells, T cells, Antibodies, Cytokines)
add_e2e_test(e2e_test_brain_immune_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_brain_immune_pipeline.cpp
)

# Distributed Immune Response E2E Test (Brain-Swarm Integration)
add_e2e_test(e2e_test_distributed_immune_response
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_distributed_immune_response.cpp
)

# Brain Immune-Plasticity Integration Pipeline E2E Test
add_e2e_test(e2e_test_brain_immune_plasticity_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_brain_immune_plasticity_pipeline.cpp
)

# Brain Immune-Emotion Integration Pipeline E2E Test
add_e2e_test(e2e_test_brain_immune_emotion_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_brain_immune_emotion_pipeline.cpp
)

# Memory-Immune Integration Pipeline E2E Test
add_e2e_test(e2e_test_memory_immune_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_memory_immune_pipeline.cpp
)

# Immune Enhancement Pipeline E2E Test (Trained Immunity, Complement, Treg, Exhaustion,
# Tolerance, Vaccine, Mucosal, Persistence)
add_e2e_test(e2e_test_immune_enhancement_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_immune_enhancement_pipeline.cpp
)

# Enhanced Wellbeing Pipeline E2E Test (Substrate, Sleep, Mental Health, Immune Integration)
add_e2e_test(e2e_test_wellbeing_enhanced_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_wellbeing_enhanced_pipeline.cpp
)

# FEP Bridges Pipeline E2E Test (Free Energy Principle Integration)
add_e2e_test(e2e_test_fep_bridges_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_fep_bridges_pipeline.cpp
)

# FEP Orchestrator Pipeline E2E Test (Central FEP Bridge Coordination)
add_e2e_test(e2e_test_fep_orchestrator_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_fep_orchestrator_pipeline.cpp
)

# Coordinator Pipeline E2E Test (6 Coordinator/Orchestrator Integration)
add_e2e_test(e2e_test_coordinator_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_coordinator_pipeline.cpp
)

# Core Directives Pipeline E2E Test (Asimov's Laws, Golden Rule, Combinatorial Harm)
add_e2e_test(e2e_test_core_directives_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_core_directives_pipeline.cpp
)

# Medulla Oblongata Pipeline E2E Test (Brainstem Autonomic Regulation)
add_e2e_test(e2e_test_medulla_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_medulla_pipeline.cpp
)

# Bridge Pipelines E2E Test (Sleep + Immune Bridges Integration)
add_executable(e2e_test_bridge_pipelines
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_bridge_pipelines.cpp
)
target_link_libraries(e2e_test_bridge_pipelines
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(e2e_test_bridge_pipelines
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/swarm/sleep
        ${CMAKE_SOURCE_DIR}/src/swarm/immune
        ${CMAKE_SOURCE_DIR}/src/security/sleep
        ${CMAKE_SOURCE_DIR}/src/cognitive
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/utils/thread
)
add_test(NAME e2e_test_bridge_pipelines COMMAND e2e_test_bridge_pipelines)
set_tests_properties(e2e_test_bridge_pipelines PROPERTIES
    LABELS "e2e"
    TIMEOUT 180
)

# Perception-Cortical Bridge E2E Test
add_executable(e2e_test_perception_cortical
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_perception_cortical.cpp
)
target_link_libraries(e2e_test_perception_cortical
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(e2e_test_perception_cortical
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/perception
        ${CMAKE_SOURCE_DIR}/src/perception/cortical
        ${CMAKE_SOURCE_DIR}/src/core/cortical_columns
)
add_test(NAME e2e_test_perception_cortical COMMAND e2e_test_perception_cortical)
set_tests_properties(e2e_test_perception_cortical PROPERTIES
    LABELS "e2e"
    TIMEOUT 180
)

# ============================================================================
# Plasticity Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for triplet STDP and plasticity orchestrator
# WHY:  Verify complete plasticity pipeline functionality
# HOW:  Test synaptic plasticity, sleep integration, immune integration
# ============================================================================
add_executable(e2e_test_plasticity_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_plasticity_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
)
target_link_libraries(e2e_test_plasticity_pipeline
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_include_directories(e2e_test_plasticity_pipeline
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(e2e_test_plasticity_pipeline PRIVATE cxx_std_17)
add_test(NAME e2e_test_plasticity_pipeline COMMAND e2e_test_plasticity_pipeline)
set_tests_properties(e2e_test_plasticity_pipeline PROPERTIES
    LABELS "e2e;plasticity"
    TIMEOUT 180
)

# ============================================================================
# Neural Plasticity Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for neural-plasticity integration (axon/neuron/dendrite)
# WHY:  Verify complete spike-to-plasticity pipeline
# HOW:  Test neuron dynamics, spike cascade, bidirectional sync
# ============================================================================
add_executable(e2e_test_neural_plasticity_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_neural_plasticity_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
)
target_link_libraries(e2e_test_neural_plasticity_pipeline
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_include_directories(e2e_test_neural_plasticity_pipeline
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(e2e_test_neural_plasticity_pipeline PRIVATE cxx_std_17)
add_test(NAME e2e_test_neural_plasticity_pipeline COMMAND e2e_test_neural_plasticity_pipeline)
set_tests_properties(e2e_test_neural_plasticity_pipeline PROPERTIES
    LABELS "e2e;plasticity;neural"
    TIMEOUT 180
)

# ============================================================================
# Master Connectivity E2E Test
# ============================================================================
# WHAT: E2E tests verifying full NIMCP brain connectivity across all layers
# WHY:  Ensure every component communicates with every other relevant component
# HOW:  Test neural substrate, thalamic/perception, Broca's region, cognitive,
#       brain infrastructure, and cross-layer integration
# ============================================================================
add_executable(e2e_test_master_connectivity
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_master_connectivity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
)
target_link_libraries(e2e_test_master_connectivity
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_include_directories(e2e_test_master_connectivity
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(e2e_test_master_connectivity PRIVATE cxx_std_17)
add_test(NAME e2e_test_master_connectivity COMMAND e2e_test_master_connectivity)
set_tests_properties(e2e_test_master_connectivity PROPERTIES
    LABELS "e2e;connectivity;master"
    TIMEOUT 300
)

# ============================================================================
# Cortical Substrate Bridge Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for cortical substrate bridge integration
# WHY:  Verify metabolic/thermal substrate modulates cortical processing
# HOW:  Test sensory-to-cortical pipelines with substrate state effects
# ============================================================================
add_executable(e2e_test_cortical_substrate_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_cortical_substrate_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
)
target_link_libraries(e2e_test_cortical_substrate_pipeline
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_include_directories(e2e_test_cortical_substrate_pipeline
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(e2e_test_cortical_substrate_pipeline PRIVATE cxx_std_17)
add_test(NAME e2e_test_cortical_substrate_pipeline COMMAND e2e_test_cortical_substrate_pipeline)
set_tests_properties(e2e_test_cortical_substrate_pipeline PROPERTIES
    LABELS "e2e;cortical;substrate"
    TIMEOUT 180
)

# ============================================================================
# Substrate Bridges Pipeline E2E Test
# ============================================================================
# WHAT: Comprehensive E2E tests for all substrate bridges across NIMCP
# WHY:  Verify complete substrate-cognitive-neural integration and stress scenarios
# HOW:  Test cognitive pipeline, neural pipeline, stress scenarios, and full brain integration
#
# TEST COVERAGE (16 tests):
# - Cognitive Pipeline (5): Full cognitive cycle, emotion-cognition interaction,
#   introspection monitoring, ToM processing, memory consolidation recovery
# - Neural Pipeline (4): Spike processing, signal propagation, plasticity learning,
#   glial support during sustained activity
# - Stress Scenarios (4): Low ATP cascade, hyperthermia effects, recovery sequence,
#   gradual degradation simulation
# - Full Brain Integration (3): Sensory-to-motor pathway, learning/adaptation cycle,
#   multi-modal processing
# ============================================================================
add_executable(e2e_test_substrate_bridges_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_substrate_bridges_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
)
target_link_libraries(e2e_test_substrate_bridges_pipeline
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_include_directories(e2e_test_substrate_bridges_pipeline
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(e2e_test_substrate_bridges_pipeline PRIVATE cxx_std_17)
add_test(NAME e2e_test_substrate_bridges_pipeline COMMAND e2e_test_substrate_bridges_pipeline)
set_tests_properties(e2e_test_substrate_bridges_pipeline PROPERTIES
    LABELS "e2e;substrate;bridges"
    TIMEOUT 300
)

# ============================================================================
# Test Execution Instructions
# ============================================================================

# Training-Logic Pipeline E2E Test (Training-Logic Bridge Integration)
add_e2e_test(e2e_test_training_logic_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_training_logic_pipeline.cpp
)

# Cognitive-Training Pipeline E2E Test (Cognitive-Training Bridge Integration)
add_e2e_test(e2e_test_cognitive_training_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_cognitive_training_pipeline.cpp
)

# Perception-Training Pipeline E2E Test (Perception-Training Bridge Integration)
add_e2e_test(e2e_test_perception_training_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_perception_training_pipeline.cpp
)

# Cortical-Training Pipeline E2E Test (Cortical-Training Bridge Integration)
add_e2e_test(e2e_test_cortical_training_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_cortical_training_pipeline.cpp
)

message(STATUS "")
message(STATUS "E2E Test Execution:")
message(STATUS "  All E2E tests:   ctest -L e2e")
message(STATUS "  Verbose:         ctest -L e2e -V")
message(STATUS "  Single test:     ./e2e_test_brain_pipeline")
message(STATUS "")
message(STATUS "===============================================================================")
message(STATUS "")
