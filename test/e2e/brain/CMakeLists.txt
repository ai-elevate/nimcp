# ============================================================================
# Brain E2E Tests
# ============================================================================
# WHAT: End-to-end tests for brain knowledge graph module wiring
# WHY:  Verify complete module wiring workflows with KG integration
# HOW:  Test full lifecycle, multi-module subsystems, and dynamic reconfiguration
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping Brain E2E tests")
    return()
endif()

# ============================================================================
# KG Module Wiring E2E Test
# ============================================================================
# WHAT: End-to-end tests for KG module wiring with brain KG integration
# WHY:  Verify complete module wiring workflows
# HOW:  Test full lifecycle, multi-module subsystems, and dynamic reconfiguration

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_kg_module_wiring_e2e.cpp")
    message(STATUS "Adding KG module wiring E2E tests...")
    add_executable(e2e_test_kg_module_wiring
        ${CMAKE_SOURCE_DIR}/test/e2e/e2e_test_framework.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_kg_module_wiring_e2e.cpp
    )

    target_link_libraries(e2e_test_kg_module_wiring
        PRIVATE
            nimcp
            GTest::GTest
            GTest::Main
            Threads::Threads
            m
    )

    target_include_directories(e2e_test_kg_module_wiring
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/test/e2e
    )

    target_compile_features(e2e_test_kg_module_wiring PRIVATE cxx_std_17)

    target_compile_definitions(e2e_test_kg_module_wiring PRIVATE
        NIMCP_TESTING
        NIMCP_E2E_TESTING
    )

    add_test(NAME e2e_test_kg_module_wiring COMMAND e2e_test_kg_module_wiring)

    set_tests_properties(e2e_test_kg_module_wiring PROPERTIES
        LABELS "e2e;brain;kg;wiring"
        TIMEOUT 120
        RUN_SERIAL TRUE
    )

    message(STATUS "  Added E2E test: e2e_test_kg_module_wiring")
endif()
