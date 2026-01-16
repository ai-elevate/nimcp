# ============================================================================
# Brain Integration Tests
# ============================================================================
# WHAT: Integration tests for brain knowledge graph module wiring
# WHY:  Verify module wiring integrates correctly with brain KG and bio-async
# HOW:  Test module registration, topology building, and query integration
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping Brain Integration tests")
    return()
endif()

# ============================================================================
# KG Module Wiring Integration Test
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_kg_module_wiring_integration.cpp")
    message(STATUS "Adding KG module wiring Integration tests...")
    add_executable(integration_test_kg_module_wiring
        ${CMAKE_CURRENT_SOURCE_DIR}/test_kg_module_wiring_integration.cpp
    )

    target_link_libraries(integration_test_kg_module_wiring
        PRIVATE
            nimcp
            GTest::GTest
            GTest::Main
            Threads::Threads
            m
    )

    target_include_directories(integration_test_kg_module_wiring
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    target_compile_features(integration_test_kg_module_wiring PRIVATE cxx_std_17)

    target_compile_definitions(integration_test_kg_module_wiring PRIVATE
        NIMCP_TESTING
    )

    add_test(NAME integration_test_kg_module_wiring COMMAND integration_test_kg_module_wiring)

    set_tests_properties(integration_test_kg_module_wiring PROPERTIES
        LABELS "integration;brain;kg;wiring"
        TIMEOUT 60
    )

    message(STATUS "  Added Integration test: integration_test_kg_module_wiring")
endif()
