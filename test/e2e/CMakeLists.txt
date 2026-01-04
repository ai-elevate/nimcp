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

# Recursive Cognition Pipeline E2E Test (RLM-inspired recursive processing)
add_e2e_test(e2e_test_recursive_cognition_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_recursive_cognition_pipeline.cpp
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

# Asymmetric Swarm E2E Test (Role-Based Brain Configurations)
add_e2e_test(e2e_test_asymmetric_swarm
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_asymmetric_swarm.cpp
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

# Game Theory Pipeline E2E Test (Auctions, Bargaining, Credit Assignment)
add_e2e_test(e2e_test_game_theory_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_game_theory_pipeline.cpp
)

# Parietal Lobe Pipeline E2E Test (Mathematical/Scientific Reasoning)
add_e2e_test(e2e_test_parietal_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_parietal_pipeline.cpp
)

# Code Immune Pipeline E2E Test (Self-Healing: Crash -> Fix -> Recompile -> Hot-Inject)
add_e2e_test(e2e_test_code_immune_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_code_immune_pipeline.cpp
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
# DISABLED: Test file uses non-existent e2e_test_framework.h and E2ETestBase
# add_executable(e2e_test_neural_plasticity_pipeline
#     ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_neural_plasticity_pipeline.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
# )
# target_link_libraries(e2e_test_neural_plasticity_pipeline
#     PRIVATE
#         nimcp
#         GTest::gtest
#         GTest::gtest_main
#         Threads::Threads
#         m
# )
# target_include_directories(e2e_test_neural_plasticity_pipeline
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
# )
# target_compile_features(e2e_test_neural_plasticity_pipeline PRIVATE cxx_std_17)
# add_test(NAME e2e_test_neural_plasticity_pipeline COMMAND e2e_test_neural_plasticity_pipeline)
# set_tests_properties(e2e_test_neural_plasticity_pipeline PROPERTIES
#     LABELS "e2e;plasticity;neural"
#     TIMEOUT 180
# )

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
# DISABLED: Uses non-existent substrate API functions (substrate_set_atp_level, etc.)
# add_executable(e2e_test_substrate_bridges_pipeline
#     ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_substrate_bridges_pipeline.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
# )
# target_link_libraries(e2e_test_substrate_bridges_pipeline
#     PRIVATE
#         nimcp
#         GTest::gtest
#         GTest::gtest_main
#         Threads::Threads
#         m
# )
# target_include_directories(e2e_test_substrate_bridges_pipeline
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
# )
# target_compile_features(e2e_test_substrate_bridges_pipeline PRIVATE cxx_std_17)
# add_test(NAME e2e_test_substrate_bridges_pipeline COMMAND e2e_test_substrate_bridges_pipeline)
# set_tests_properties(e2e_test_substrate_bridges_pipeline PROPERTIES
#     LABELS "e2e;substrate;bridges"
#     TIMEOUT 300
# )

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

# Unified Training Integration E2E Test (All Perception/Cortical Integrations)
add_e2e_test(e2e_test_unified_training_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_unified_training_integration.cpp
)

# CNN-Cortex Bridge E2E Test (CNN Training with A/V Cortex Feature Extraction)
add_e2e_test(e2e_test_cnn_cortex_bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_cnn_cortex_bridge.cpp
)

# CNN Backpropagation E2E Test (CNN Training Convergence and Inference)
add_e2e_test(e2e_test_cnn_backprop
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_cnn_backprop.cpp
)

# SNN Pipeline E2E Test (Spiking Neural Network Full Integration)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_snn_pipeline.cpp)
    add_executable(e2e_test_snn_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_snn_pipeline.cpp
    )
    target_include_directories(e2e_test_snn_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/snn
    )
    target_link_libraries(e2e_test_snn_pipeline PRIVATE
        nimcp_snn
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_snn_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_snn_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;snn"
            TIMEOUT 120
    )
    message(STATUS "  Added SNN E2E test: e2e_test_snn_pipeline")
endif()

# SNN Bridge Pipeline E2E Test
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_snn_bridge_pipeline.cpp)
    add_executable(e2e_test_snn_bridge_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_snn_bridge_pipeline.cpp
    )
    target_include_directories(e2e_test_snn_bridge_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/snn
    )
    target_link_libraries(e2e_test_snn_bridge_pipeline PRIVATE
        nimcp_snn
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_snn_bridge_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_snn_bridge_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;snn;bridges"
            TIMEOUT 180
    )
    message(STATUS "  Added SNN Bridge E2E test: e2e_test_snn_bridge_pipeline")
endif()

# ============================================================================
# Quantum Bridges Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for quantum-enhanced cognitive bridges
# WHY:  Verify quantum acceleration works in complete cognitive pipelines
# HOW:  Test attention, curiosity, emotion, executive, WM, BCM, thalamic integration
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_quantum_bridges_pipeline.cpp)
    add_executable(e2e_test_quantum_bridges_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_quantum_bridges_pipeline.cpp
    )
    target_include_directories(e2e_test_quantum_bridges_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/plasticity/attention
        ${CMAKE_SOURCE_DIR}/include/cognitive/curiosity
        ${CMAKE_SOURCE_DIR}/include/cognitive/emotion
        ${CMAKE_SOURCE_DIR}/include/cognitive/executive
        ${CMAKE_SOURCE_DIR}/include/cognitive/memory
        ${CMAKE_SOURCE_DIR}/include/core/logic
        ${CMAKE_SOURCE_DIR}/include/plasticity/bcm
        ${CMAKE_SOURCE_DIR}/include/middleware/routing
    )
    target_link_libraries(e2e_test_quantum_bridges_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_quantum_bridges_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_quantum_bridges_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;quantum;bridges"
            TIMEOUT 180
    )
    message(STATUS "  Added Quantum Bridges E2E test: e2e_test_quantum_bridges_pipeline")
endif()

# ============================================================================
# Pink Noise Enhancement Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for 9 pink noise enhancement modules
# WHY:  Verify complete neural noise pipeline with all enhancements
# HOW:  Test multiscale, correlated, criticality, quantum, immune, sleep,
#       SIMD, monitor, and spatial noise in integrated brain scenarios
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_pink_noise_pipeline.cpp)
    add_executable(e2e_test_pink_noise_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_pink_noise_pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    )
    target_include_directories(e2e_test_pink_noise_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/plasticity/noise
    )
    target_link_libraries(e2e_test_pink_noise_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_pink_noise_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_pink_noise_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;plasticity;noise;pink"
            TIMEOUT 300
    )
    message(STATUS "  Added Pink Noise E2E test: e2e_test_pink_noise_pipeline")
endif()

# ============================================================================
# Enhanced Curiosity Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for Enhanced Curiosity System (10 enhancements)
# WHY:  Verify complete curiosity-driven exploration pipeline
# HOW:  Test boredom, interest, types, anxiety, social, meta, contagion,
#       surprise, fatigue, counterfactual in integrated scenarios
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_curiosity_enhanced_pipeline.cpp)
    add_executable(e2e_test_curiosity_enhanced_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_curiosity_enhanced_pipeline.cpp
    )
    target_include_directories(e2e_test_curiosity_enhanced_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/cognitive/curiosity
    )
    target_link_libraries(e2e_test_curiosity_enhanced_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_curiosity_enhanced_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_curiosity_enhanced_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;cognitive;curiosity;enhanced"
            TIMEOUT 300
    )
    message(STATUS "  Added Enhanced Curiosity E2E test: e2e_test_curiosity_enhanced_pipeline")
endif()

# ============================================================================
# Phase 1 Bridges Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for Phase 1 bridges (dendrite, synapse, axon, glial-immune)
# WHY:  Verify complete bridge pipeline in real-world scenarios
# HOW:  Test learning, immune challenge, myelination, coordinated operation
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_phase1_bridges_pipeline.cpp)
    add_executable(e2e_test_phase1_bridges_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_phase1_bridges_pipeline.cpp
    )
    target_include_directories(e2e_test_phase1_bridges_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/plasticity/bridges
        ${CMAKE_SOURCE_DIR}/include/glial/immune
    )
    target_link_libraries(e2e_test_phase1_bridges_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_phase1_bridges_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_phase1_bridges_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;plasticity;bridges;phase1"
            TIMEOUT 300
    )
    message(STATUS "  Added Phase 1 Bridges E2E test: e2e_test_phase1_bridges_pipeline")
endif()

# ============================================================================
# Subcortical Brain Structures Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for basal ganglia action selection and reinforcement learning
# WHY:  Verify complete subcortical pipeline in realistic decision scenarios
# HOW:  Test action selection, RL learning, habit formation, pathway dynamics
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_subcortical_pipeline.cpp)
    add_executable(e2e_test_subcortical_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_subcortical_pipeline.cpp
    )
    target_include_directories(e2e_test_subcortical_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/core/brain/subcortical
    )
    target_link_libraries(e2e_test_subcortical_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_subcortical_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_subcortical_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;core;subcortical;basal_ganglia"
            TIMEOUT 180
    )
    message(STATUS "  Added Subcortical E2E test: e2e_test_subcortical_pipeline")
endif()

# ============================================================================
# Thalamus Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for thalamic relay and attention gating pipelines
# WHY:  Verify complete thalamus-cortex relay in realistic scenarios
# HOW:  Test visual/motor relay, attention modulation, arousal transitions
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_thalamus_pipeline.cpp)
    add_executable(e2e_test_thalamus_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_thalamus_pipeline.cpp
    )
    target_include_directories(e2e_test_thalamus_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/core/brain/subcortical
    )
    target_link_libraries(e2e_test_thalamus_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_thalamus_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_thalamus_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;core;subcortical;thalamus"
            TIMEOUT 180
    )
    message(STATUS "  Added Thalamus E2E test: e2e_test_thalamus_pipeline")
endif()

# ============================================================================
# Amygdala Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for amygdala fear conditioning and emotion pipelines
# WHY:  Verify complete fear learning, extinction, and renewal in realistic scenarios
# HOW:  Test Pavlovian conditioning, contextual fear, neuromodulation effects
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_amygdala_pipeline.cpp)
    add_executable(e2e_test_amygdala_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_amygdala_pipeline.cpp
    )
    target_include_directories(e2e_test_amygdala_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/core/brain/subcortical
    )
    target_link_libraries(e2e_test_amygdala_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_amygdala_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_amygdala_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;core;subcortical;amygdala"
            TIMEOUT 180
    )
    message(STATUS "  Added Amygdala E2E test: e2e_test_amygdala_pipeline")
endif()

# ============================================================================
# Dragonfly Hunting Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for complete dragonfly hunting pipeline
# WHY:  Verify 95% interception success rate with realistic physics
# HOW:  Simulate full hunt from detection to interception
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_dragonfly_hunting_pipeline.cpp)
    add_executable(e2e_test_dragonfly_hunting_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_dragonfly_hunting_pipeline.cpp
    )
    target_include_directories(e2e_test_dragonfly_hunting_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/dragonfly
    )
    target_link_libraries(e2e_test_dragonfly_hunting_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_dragonfly_hunting_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_dragonfly_hunting_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;dragonfly;hunting"
            TIMEOUT 300
    )
    message(STATUS "  Added Dragonfly Hunting E2E test: e2e_test_dragonfly_hunting_pipeline")
endif()

# ============================================================================
# Dragonfly Multi-Target E2E Test
# ============================================================================
# WHAT: E2E tests for dragonfly handling multiple simultaneous targets
# WHY:  Real scenarios have multiple prey items - CSTMD1 must select one
# HOW:  Present multiple targets, verify winner-take-all selection
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_dragonfly_multi_target.cpp)
    add_executable(e2e_test_dragonfly_multi_target
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_dragonfly_multi_target.cpp
    )
    target_include_directories(e2e_test_dragonfly_multi_target PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/dragonfly
    )
    target_link_libraries(e2e_test_dragonfly_multi_target PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_dragonfly_multi_target PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_dragonfly_multi_target
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;dragonfly;multi-target"
            TIMEOUT 300
    )
    message(STATUS "  Added Dragonfly Multi-Target E2E test: e2e_test_dragonfly_multi_target")
endif()

# ============================================================================
# Basal Ganglia Integration E2E Test
# ============================================================================
# WHAT: Test enhanced BG integration with brain lifecycle
# WHY:  Validate action selection, reward processing, emotional integration
# HOW:  Create brain → init BG → select actions → process rewards
# ============================================================================
add_e2e_test(e2e_test_basal_ganglia_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_basal_ganglia_pipeline.cpp
)

# ============================================================================
# Quantum Reasoning Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for quantum reasoning with brain integration
# WHY:  Verify Grover-accelerated SAT solving in cognitive pipelines
# HOW:  Test transitive/disjunctive reasoning, conflict detection, planning
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_quantum_reasoning_pipeline.cpp)
    add_executable(e2e_test_quantum_reasoning_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_quantum_reasoning_pipeline.cpp
    )
    target_include_directories(e2e_test_quantum_reasoning_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/cognitive/reasoning
        ${CMAKE_SOURCE_DIR}/include/core/brain/inference
    )
    target_link_libraries(e2e_test_quantum_reasoning_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_quantum_reasoning_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_quantum_reasoning_pipeline
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;quantum;reasoning"
            TIMEOUT 180
    )
    message(STATUS "  Added Quantum Reasoning E2E test: e2e_test_quantum_reasoning_pipeline")
endif()

# ============================================================================
# Quantum Language Processing Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for quantum-accelerated language production
# WHY:  Verify Broca quantum bridge in full language pipelines
# HOW:  Test word retrieval, syntax planning, phoneme optimization
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_quantum_language_processing.cpp)
    add_executable(e2e_test_quantum_language_processing
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_quantum_language_processing.cpp
    )
    target_include_directories(e2e_test_quantum_language_processing PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/core/brain/regions/broca
    )
    target_link_libraries(e2e_test_quantum_language_processing PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_quantum_language_processing PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_quantum_language_processing
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;quantum;language;broca"
            TIMEOUT 180
    )
    message(STATUS "  Added Quantum Language E2E test: e2e_test_quantum_language_processing")
endif()

# ============================================================================
# Auditory Processing Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for auditory cortex processing pipeline
# WHY:  Verify complete audio sensory-to-cognitive flow
# HOW:  Test sound localization, speech processing, auditory attention
# ============================================================================
add_e2e_test(e2e_test_auditory_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_auditory_pipeline.cpp
)

# ============================================================================
# Cerebellum Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for cerebellar motor coordination pipeline
# WHY:  Verify motor learning, timing, and coordination
# HOW:  Test forward models, error correction, predictive timing
# ============================================================================
add_e2e_test(e2e_test_cerebellum_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_cerebellum_pipeline.cpp
)

# ============================================================================
# Dragonfly Multimodal E2E Test
# ============================================================================
# WHAT: E2E tests for dragonfly multi-sensory integration
# WHY:  Verify visual-audio-motor coordination for hunting
# HOW:  Test cross-modal binding, attention switching, pursuit
# ============================================================================
add_e2e_test(e2e_test_dragonfly_multimodal
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_dragonfly_multimodal.cpp
)

# ============================================================================
# Dragonfly Main Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for complete dragonfly hunting pipeline
# WHY:  Verify target detection, tracking, interception
# HOW:  Test TSDN, CSTMD1, prediction, motor output
# ============================================================================
add_e2e_test(e2e_test_dragonfly_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_dragonfly_pipeline.cpp
)

# ============================================================================
# Hippocampus Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for hippocampal memory and navigation
# WHY:  Verify spatial memory, episodic encoding, consolidation
# HOW:  Test place cells, grid cells, memory formation
# ============================================================================
add_e2e_test(e2e_test_hippocampus_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_hippocampus_pipeline.cpp
)

# ============================================================================
# Limbic System Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for limbic emotional processing
# WHY:  Verify emotion-cognition integration, motivation, reward
# HOW:  Test amygdala, reward circuits, emotional memory
# ============================================================================
add_e2e_test(e2e_test_limbic_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_limbic_pipeline.cpp
)

# ============================================================================
# Motor Control Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for motor control and execution
# WHY:  Verify motor planning, execution, feedback control
# HOW:  Test motor cortex, basal ganglia, motor learning
# ============================================================================
add_e2e_test(e2e_test_motor_control_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_motor_control_pipeline.cpp
)

# ============================================================================
# Prefrontal Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for prefrontal executive functions
# WHY:  Verify decision making, working memory, planning
# HOW:  Test executive control, goal maintenance, cognitive flexibility
# ============================================================================
add_e2e_test(e2e_test_prefrontal_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_prefrontal_pipeline.cpp
)

# ============================================================================
# Somatosensory Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for somatosensory processing
# WHY:  Verify touch, proprioception, body awareness
# HOW:  Test tactile processing, body schema, sensorimotor integration
# ============================================================================
add_e2e_test(e2e_test_somatosensory_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_somatosensory_pipeline.cpp
)

# ============================================================================
# Visual Pipeline E2E Test
# ============================================================================
# WHAT: E2E tests for visual processing pipeline
# WHY:  Verify visual cortex hierarchy, object recognition, attention
# HOW:  Test V1-V4, IT cortex, visual attention, scene understanding
# ============================================================================
add_e2e_test(e2e_test_visual_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_visual_pipeline.cpp
)

# ============================================================================
# Swarm Memory Consolidation E2E Test (2025-12-31)
# ============================================================================
# WHAT: E2E tests for swarm memory consolidation workflow
# WHY:  Verify complete memory lifecycle with correct struct members
# HOW:  Test storage, consolidation, distribution, forgetting
# REGRESSION: Tests NimcpHippocampusNode->memory_count, payload_size usage
# ============================================================================
add_e2e_test(e2e_test_swarm_memory_consolidation
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_swarm_memory_consolidation.cpp
)

# ============================================================================
# Brain Initialization Subsystems E2E Test (2025-12-31)
# ============================================================================
# WHAT: E2E tests for brain subsystem initialization
# WHY:  Verify correct struct members and enum values during init
# HOW:  Test hypothalamus, wellbeing, collective cognition, power management
# REGRESSION: Tests wellbeing_monitoring_enabled, POWER_EVENT_BATTERY_CRITICAL
# ============================================================================
add_e2e_test(e2e_test_brain_initialization_subsystems
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_brain_initialization_subsystems.cpp
)

# ============================================================================
# Wiring Diagram Pipeline E2E Test (2026-01-03)
# ============================================================================
# WHAT: E2E test for KG-based runtime module wiring system
# WHY:  Verify complete wiring pipeline from JSONL loading to handler registration
# HOW:  Test wiring loading, discovery, callback invocation, and persistence
add_e2e_test(e2e_test_wiring_diagram_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_wiring_diagram_pipeline.cpp
)

# ============================================================================
# Omnidirectional Inference Pipeline E2E Test (2026-01-04)
# ============================================================================
# WHAT: E2E test for complete omnidirectional inference pipeline
# WHY:  Verify bidirectional JEPA, Hopfield memory, predictive hierarchy,
#       temporal replay, and all cortical bridges work together
# HOW:  Test full perception-to-action pipeline with all prediction directions
add_e2e_test(e2e_test_omni_inference_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_omni_inference_pipeline.cpp
)

# ============================================================================
# GPU Kernel Pipeline E2E Tests (2025-12-31)
# ============================================================================
# WHAT: E2E tests for GPU training, inference, SNN, and LNN pipelines
# WHY:  Verify complete GPU kernel operations in realistic neural workloads
# HOW:  Test full training loops, inference batches, SNN spikes, LNN dynamics
# ============================================================================

# GPU Neural Network Training Pipeline (full forward/backward pass)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_nn_training_pipeline.cpp)
    add_e2e_test(e2e_test_gpu_nn_training_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_nn_training_pipeline.cpp
    )
    set_tests_properties(e2e_test_gpu_nn_training_pipeline PROPERTIES
        LABELS "e2e;gpu;training"
        TIMEOUT 180
    )
endif()

# GPU Inference Pipeline (batched inference with quantization)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_inference_pipeline.cpp)
    add_e2e_test(e2e_test_gpu_inference_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_inference_pipeline.cpp
    )
    set_tests_properties(e2e_test_gpu_inference_pipeline PROPERTIES
        LABELS "e2e;gpu;inference"
        TIMEOUT 180
    )
endif()

# GPU SNN Pipeline (spiking neural network with STDP)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_snn_pipeline.cpp)
    add_e2e_test(e2e_test_gpu_snn_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_snn_pipeline.cpp
    )
    set_tests_properties(e2e_test_gpu_snn_pipeline PROPERTIES
        LABELS "e2e;gpu;snn"
        TIMEOUT 180
    )
endif()

# GPU LNN Pipeline (Liquid Neural Network with ODE solvers)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_lnn_pipeline.cpp)
    add_e2e_test(e2e_test_gpu_lnn_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_lnn_pipeline.cpp
    )
    set_tests_properties(e2e_test_gpu_lnn_pipeline PROPERTIES
        LABELS "e2e;gpu;lnn"
        TIMEOUT 180
    )
endif()

# GPU Training Pipeline (basic training with optimizers)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_training_pipeline.cpp)
    add_e2e_test(e2e_test_gpu_training_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_gpu_training_pipeline.cpp
    )
    set_tests_properties(e2e_test_gpu_training_pipeline PROPERTIES
        LABELS "e2e;gpu;training"
        TIMEOUT 180
    )
endif()

# ============================================================================
# Mental Health Guardian E2E Tests (Phase 10.5.1)
# ============================================================================
# WHAT: Full lifecycle tests for guardian in complete brain context
# WHY:  Verify guardian works correctly in production-like scenarios
# HOW:  Test full brain lifecycle with guardian monitoring

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_mental_health_guardian_e2e.cpp)
    add_e2e_test(e2e_test_mental_health_guardian
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_mental_health_guardian_e2e.cpp
    )
    set_tests_properties(e2e_test_mental_health_guardian PROPERTIES
        LABELS "e2e;cognitive;mental_health;guardian"
        TIMEOUT 180
    )
endif()

# ============================================================================
# GPU Broca Pipeline E2E Test
# ============================================================================
# WHAT: End-to-end tests for GPU-accelerated Broca language production pipeline
# WHY:  Validate full GPU Broca workflow from semantic intent to motor output
# HOW:  Test realistic utterance production with performance benchmarks
#
# PIPELINE STAGES:
# - Vocabulary loading -> Lexical lookup -> Phoneme encoding -> Motor planning
# ============================================================================
if(CUDAToolkit_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_broca_gpu_pipeline.cpp)
    add_executable(e2e_test_broca_gpu_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_broca_gpu_pipeline.cpp
    )
    target_include_directories(e2e_test_broca_gpu_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/gpu/cognitive
    )
    target_link_libraries(e2e_test_broca_gpu_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_broca_gpu_pipeline PRIVATE cxx_std_20)
    target_compile_definitions(e2e_test_broca_gpu_pipeline PRIVATE NIMCP_TESTING)
    add_test(NAME e2e_test_broca_gpu_pipeline COMMAND e2e_test_broca_gpu_pipeline)
    set_tests_properties(e2e_test_broca_gpu_pipeline PROPERTIES
        LABELS "e2e;gpu;broca;language"
        TIMEOUT 300
    )
    message(STATUS "  Added GPU Broca Pipeline E2E test: e2e_test_broca_gpu_pipeline")
endif()

# ============================================================================
# E2E Test: GPU Neural Substrate Pipeline
# ============================================================================
# WHAT: End-to-end test for GPU-accelerated neural substrate pipeline
# WHY:  Validate complete GPU substrate from initialization to simulation
# HOW:  Test all subsystems (axon, dendrite, myelin, neuromod, glial, metabolic)
#
# PIPELINE STAGES:
# - Full substrate initialization -> Multi-step simulation -> Performance benchmarks
# ============================================================================
if(CUDAToolkit_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_substrate_gpu_pipeline.cpp)
    add_executable(e2e_test_substrate_gpu_pipeline
        ${CMAKE_CURRENT_SOURCE_DIR}/e2e_test_substrate_gpu_pipeline.cpp
    )
    target_include_directories(e2e_test_substrate_gpu_pipeline PRIVATE
        ${E2E_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/gpu/substrate
    )
    target_link_libraries(e2e_test_substrate_gpu_pipeline PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )
    target_compile_features(e2e_test_substrate_gpu_pipeline PRIVATE cxx_std_20)
    target_compile_definitions(e2e_test_substrate_gpu_pipeline PRIVATE NIMCP_TESTING)
    add_test(NAME e2e_test_substrate_gpu_pipeline COMMAND e2e_test_substrate_gpu_pipeline)
    set_tests_properties(e2e_test_substrate_gpu_pipeline PROPERTIES
        LABELS "e2e;gpu;substrate;neural"
        TIMEOUT 300
    )
    message(STATUS "  Added GPU Substrate Pipeline E2E test: e2e_test_substrate_gpu_pipeline")
endif()

# Include GPU E2E test subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/CMakeLists.txt)
    add_subdirectory(gpu)
endif()

# Include Ternary E2E test subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ternary/CMakeLists.txt)
    add_subdirectory(ternary)
endif()

message(STATUS "")
message(STATUS "E2E Test Execution:")
message(STATUS "  All E2E tests:   ctest -L e2e")
message(STATUS "  Verbose:         ctest -L e2e -V")
message(STATUS "  Single test:     ./e2e_test_brain_pipeline")
message(STATUS "")
message(STATUS "===============================================================================")
message(STATUS "")
