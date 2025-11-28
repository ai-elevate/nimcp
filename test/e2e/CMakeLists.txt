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
    nimcp
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

# ============================================================================
# Test Execution Instructions
# ============================================================================

message(STATUS "")
message(STATUS "E2E Test Execution:")
message(STATUS "  All E2E tests:   ctest -L e2e")
message(STATUS "  Verbose:         ctest -L e2e -V")
message(STATUS "  Single test:     ./e2e_test_brain_pipeline")
message(STATUS "")
message(STATUS "===============================================================================")
message(STATUS "")
