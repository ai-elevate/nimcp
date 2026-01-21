# =============================================================================
# Unit Tests for Brain Module
# =============================================================================
# WHAT: Unit tests for core brain functionality
# WHY:  Verify correctness of brain module APIs
# HOW:  GTest-based unit tests
# =============================================================================

# KG Module Wiring Tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_kg_module_wiring.cpp")
    add_executable(unit_brain_test_kg_module_wiring
        ${CMAKE_CURRENT_SOURCE_DIR}/test_kg_module_wiring.cpp
    )

    target_link_libraries(unit_brain_test_kg_module_wiring
        PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
    )

    target_include_directories(unit_brain_test_kg_module_wiring
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/core/brain
    )

    target_compile_features(unit_brain_test_kg_module_wiring PRIVATE cxx_std_17)
    target_compile_definitions(unit_brain_test_kg_module_wiring PRIVATE NIMCP_TESTING NIMCP_UNIT_TESTING)

    gtest_discover_tests(unit_brain_test_kg_module_wiring
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;brain;kg_module_wiring"
            TIMEOUT 120
    )

    message(STATUS "  Added Unit test: unit_brain_test_kg_module_wiring")
endif()
