# ============================================================================
# Cognitive E2E Tests
# ============================================================================
# WHAT: End-to-end tests for cognitive workflows
# WHY:  Verify complete cognitive workflows: training loops, prediction
#       pipelines, counterfactual reasoning, exception handling, and policy evaluation
# HOW:  GTest-based tests for realistic scenarios
#
# TESTS INCLUDED:
# - World model workflows (DreamerV3/JEPA inspired)
# - ToM exception handling
# - WM bridges exception handling
# - Predictive bridge exception handling
# ============================================================================

# Find test sources
file(GLOB WM_E2E_SOURCES "*.cpp")

# Guard: Check if test sources exist
if(NOT WM_E2E_SOURCES)
    message(STATUS "No world model E2E tests found")
    return()
endif()

# Create test executable for each test file
foreach(TEST_FILE ${WM_E2E_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    set(TARGET_NAME "e2e_cognitive_${TEST_NAME}")

    add_executable(${TARGET_NAME} ${TEST_FILE})

    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/test
    )

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            nimcp
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            Threads::Threads
            m
    )

    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
    target_compile_definitions(${TARGET_NAME} PRIVATE NIMCP_TESTING)

    # Register test with CTest
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    set_tests_properties(${TARGET_NAME} PROPERTIES
        LABELS "e2e;cognitive;world_model"
        TIMEOUT 300
    )
endforeach()

message(STATUS "Added world model E2E tests")

# ============================================================================
# Include subdirectories
# ============================================================================
add_subdirectory(neuro_symbolic)
add_subdirectory(immune)
