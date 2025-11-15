# ============================================================================
# NIMCP Test Framework - Code Surgeon Integration
# ============================================================================
# WHAT: Build system for migrated test framework
# WHY:  Enable Code Surgeon automated testing with parallel execution
# HOW:  Discover all tests by category, build individually for parallelism
#
# STRUCTURE:
# - test/unit/        - Unit tests (per-module testing)
# - test/integration/ - Integration tests (cross-module)
# - test/e2e/         - End-to-end tests (full system)
# - test/regression/  - Regression tests (bug reproduction)
# - test/fuzz/        - Fuzzing tests (property-based)
#
# NIMCP STANDARDS:
# - All functions < 50 lines
# - Guard clauses (early returns)
# - WHAT-WHY-HOW documentation
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping test framework")
    return()
endif()

# ============================================================================
# Common Test Configuration
# ============================================================================

set(COMMON_TEST_INCLUDES
    ${NIMCP_INCLUDE_DIR}
    ${NIMCP_TEST_DIR}
    ${NIMCP_TEST_DIR}/utils
    ${NIMCP_LIB_DIR}
    ${GTEST_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

set(COMMON_TEST_LIBS
    nimcp
    GTest::GTest
    GTest::Main
    Python3::Python
    pthread
)

# ============================================================================
# Helper Function: Create Test Binary
# ============================================================================
# WHAT: Create test executable with standard NIMCP configuration
# WHY:  Ensure consistent build settings across all tests
# HOW:  Configure includes, libs, coverage, CTest registration
#
# PARAMETERS:
#   TARGET_NAME - Name of test executable
#   SOURCE_FILE - Test source file
#   CATEGORY    - Test category (unit/integration/e2e/regression/fuzz)
# ============================================================================
function(add_test_binary TARGET_NAME SOURCE_FILE CATEGORY)
    # Guard: Validate source file exists
    if(NOT EXISTS ${SOURCE_FILE})
        message(WARNING "Test source not found: ${SOURCE_FILE}")
        return()
    endif()

    # Create executable
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    # Configure includes
    target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_TEST_INCLUDES})

    # Enable C++20 for designated initializers
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

    # Enable test-only APIs
    target_compile_definitions(${TARGET_NAME} PRIVATE NIMCP_TESTING)

    # Add Python module source
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/python/nimcp_module.c
    )

    # Link libraries
    target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_TEST_LIBS})

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

    # Register with CTest
    add_test(NAME ${TARGET_NAME}
             COMMAND ${TARGET_NAME}
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Set test properties
    set_tests_properties(${TARGET_NAME} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python"
        LABELS ${CATEGORY}
    )
endfunction()

# ============================================================================
# Helper Function: Discover Tests in Category (Recursive)
# ============================================================================
# WHAT: Recursively find all test files in a category directory tree
# WHY:  Support modular organization of tests by subdirectory
# HOW:  GLOB_RECURSE for test files, create executable for each
#
# PARAMETERS:
#   CATEGORY - Test category subdirectory name (unit/integration/regression)
# ============================================================================
function(discover_category_tests CATEGORY)
    # Build path to category directory
    set(CATEGORY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CATEGORY})

    # Guard: Check if category directory exists
    if(NOT EXISTS ${CATEGORY_DIR})
        message(STATUS "Category directory not found: ${CATEGORY}")
        return()
    endif()

    # Recursively find all test files in subdirectories
    file(GLOB_RECURSE TEST_FILES "${CATEGORY_DIR}/*.cpp" "${CATEGORY_DIR}/*.c")

    # Guard: No tests found
    if(NOT TEST_FILES)
        message(STATUS "No tests found in ${CATEGORY}/")
        return()
    endif()

    # Create executable for each test file
    foreach(TEST_FILE ${TEST_FILES})
        # Extract test name from filename
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

        # Get relative path from category dir to determine module
        file(RELATIVE_PATH REL_PATH ${CATEGORY_DIR} ${TEST_FILE})
        get_filename_component(MODULE_PATH ${REL_PATH} DIRECTORY)

        # Create unique target name: category_module_testname
        # Replace / with _ in module path for valid target name
        string(REPLACE "/" "_" MODULE_NAME "${MODULE_PATH}")
        if(MODULE_NAME)
            set(TARGET_NAME "${CATEGORY}_${MODULE_NAME}_${TEST_NAME}")
        else()
            set(TARGET_NAME "${CATEGORY}_${TEST_NAME}")
        endif()

        # Create test binary
        add_test_binary(${TARGET_NAME} ${TEST_FILE} ${CATEGORY})
    endforeach()

    # Report
    list(LENGTH TEST_FILES TEST_COUNT)
    message(STATUS "  ${CATEGORY}: ${TEST_COUNT} tests (across all modules)")
endfunction()

# ============================================================================
# Discover and Build All Tests
# ============================================================================

message(STATUS "")
message(STATUS "===============================================================================")
message(STATUS "NIMCP Test Framework - Code Surgeon Integration")
message(STATUS "===============================================================================")

# Discover tests in each category
discover_category_tests(unit)
discover_category_tests(integration)
discover_category_tests(e2e)
discover_category_tests(regression)
discover_category_tests(fuzz)

# Enable CTest
enable_testing()

# ============================================================================
# Test Execution Instructions
# ============================================================================

message(STATUS "")
message(STATUS "Test Execution:")
message(STATUS "  All tests:       ctest -j\$(nproc)")
message(STATUS "  Unit only:       ctest -L unit -j\$(nproc)")
message(STATUS "  Integration:     ctest -L integration -j\$(nproc)")
message(STATUS "  E2E only:        ctest -L e2e")
message(STATUS "  Regression:      ctest -L regression")
message(STATUS "  Verbose:         ctest -V")
message(STATUS "")
message(STATUS "Code Surgeon Integration:")
message(STATUS "  ./tools/code_surgeon/code_surgeon.py --mode test-only")
message(STATUS "  ./tools/code_surgeon/code_surgeon.py --mode full")
message(STATUS "===============================================================================")
message(STATUS "")
