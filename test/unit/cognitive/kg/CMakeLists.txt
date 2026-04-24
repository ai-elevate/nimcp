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
