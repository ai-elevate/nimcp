# =============================================================================
# Unit Tests for KG Integration Waves
# =============================================================================
# WHAT: Verify each KG retrofit wave wires the expected nodes/edges into
#       brain->internal_kg and that idempotent re-init doesn't duplicate.
# WHY:  Catches statue-regressions (comments where node adds should be) and
#       catches cross-module name drift (see kg-node-naming-registry.md).
# HOW:  GTest. Each test creates a minimal brain, exercises one wave's code
#       path, then asserts via brain_kg_find_node.
# =============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave1_statue_finish.cpp")
    add_executable(unit_cognitive_kg_test_wave1_statue_finish
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave1_statue_finish.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave1_statue_finish
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave1_statue_finish
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave1_statue_finish PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave1_statue_finish PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave1_statue_finish
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave1"
            TIMEOUT 120
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave1_statue_finish")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave2_region_events.cpp")
    add_executable(unit_cognitive_kg_test_wave2_region_events
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave2_region_events.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave2_region_events
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave2_region_events
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave2_region_events PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave2_region_events PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave2_region_events
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave2"
            TIMEOUT 120
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave2_region_events")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave3_unwired_regions.cpp")
    add_executable(unit_cognitive_kg_test_wave3_unwired_regions
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave3_unwired_regions.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave3_unwired_regions
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave3_unwired_regions
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave3_unwired_regions PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave3_unwired_regions PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave3_unwired_regions
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave3"
            TIMEOUT 120
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave3_unwired_regions")
endif()

# -----------------------------------------------------------------------------
# Wave W5: network-type root nodes + emit functions (LNN/SNN/CNN/FNO/HNN/main)
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave5_network_types.cpp")
    add_executable(unit_cognitive_kg_test_wave5_network_types
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave5_network_types.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave5_network_types
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave5_network_types
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave5_network_types PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave5_network_types PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave5_network_types
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave5"
            TIMEOUT 120
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave5_network_types")
endif()

# -----------------------------------------------------------------------------
# Wave W6: cognitive-memory family (KG-native, bidirectional)
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave6_memory_family.cpp")
    add_executable(unit_cognitive_kg_test_wave6_memory_family
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave6_memory_family.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave6_memory_family
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave6_memory_family
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave6_memory_family
        PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave6_memory_family
        PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave6_memory_family
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave6"
            TIMEOUT 180
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave6_memory_family")
endif()

# -----------------------------------------------------------------------------
# Wave W4: peptide subsystems + per-subcortical runtime events
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave4_peptide_and_subcortical.cpp")
    add_executable(unit_cognitive_kg_test_wave4_peptide_and_subcortical
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave4_peptide_and_subcortical.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave4_peptide_and_subcortical
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave4_peptide_and_subcortical
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave4_peptide_and_subcortical
        PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave4_peptide_and_subcortical
        PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave4_peptide_and_subcortical
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave4"
            TIMEOUT 180
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave4_peptide_and_subcortical")
endif()

# -----------------------------------------------------------------------------
# Wave W7: reasoning / knowledge / symbolic-logic family (2026-04-24)
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave7_reasoning_knowledge.cpp")
    add_executable(unit_cognitive_kg_test_wave7_reasoning_knowledge
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave7_reasoning_knowledge.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave7_reasoning_knowledge
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave7_reasoning_knowledge
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave7_reasoning_knowledge
        PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave7_reasoning_knowledge
        PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave7_reasoning_knowledge
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave7"
            TIMEOUT 180
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave7_reasoning_knowledge")
endif()

# -----------------------------------------------------------------------------
# Wave W16: real runtime KG consumers (brain_decide salience bias + prior
# decision recurrence + middleware attention_focus) (2026-04-24)
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave16_real_consumers.cpp")
    add_executable(unit_cognitive_kg_test_wave16_real_consumers
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave16_real_consumers.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave16_real_consumers
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave16_real_consumers
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave16_real_consumers
        PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave16_real_consumers
        PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave16_real_consumers
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave16"
            TIMEOUT 180
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave16_real_consumers")
endif()

# -----------------------------------------------------------------------------
# Wave W17: KG consumers in brain_learn_vector (training path) (2026-04-24)
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_wave17_training_consumers.cpp")
    add_executable(unit_cognitive_kg_test_wave17_training_consumers
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wave17_training_consumers.cpp
    )

    target_link_libraries(unit_cognitive_kg_test_wave17_training_consumers
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_cognitive_kg_test_wave17_training_consumers
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_features(unit_cognitive_kg_test_wave17_training_consumers
        PRIVATE cxx_std_17)
    target_compile_definitions(unit_cognitive_kg_test_wave17_training_consumers
        PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_cognitive_kg_test_wave17_training_consumers
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;cognitive;kg;wave17"
            TIMEOUT 240
    )

    message(STATUS "  Added Unit test: unit_cognitive_kg_test_wave17_training_consumers")
endif()
