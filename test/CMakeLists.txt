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

# Define directory variables
set(NIMCP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(NIMCP_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NIMCP_LIB_DIR ${CMAKE_SOURCE_DIR}/src/lib)

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

    # Add explicit dependency on nimcp library
    add_dependencies(${TARGET_NAME} nimcp)

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
    target_link_libraries(${TARGET_NAME}
    PRIVATE
        nimcp
        GTest::GTest
        GTest::Main
        Python3::Python
        pthread
)

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
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python;LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp"
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
        # Check if test file's directory has its own CMakeLists.txt
        # If it does, skip this test file (it's managed by the subdirectory CMakeLists.txt)
        get_filename_component(TEST_DIR ${TEST_FILE} DIRECTORY)
        if(EXISTS "${TEST_DIR}/CMakeLists.txt")
            # Skip - this test is managed by subdirectory CMakeLists.txt
            continue()
        endif()

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
discover_category_tests(regression)
discover_category_tests(fuzz)

# E2E tests have their own CMakeLists.txt (uses custom build rules)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/e2e")
    message(STATUS "Adding E2E test subdirectory...")
    add_subdirectory(e2e)
endif()

# Enable CTest
enable_testing()

# ============================================================================
# Timeout Configuration for Slow Tests
# ============================================================================
# WHAT: Extend timeout for computationally intensive tests
# WHY:  Prevent false failures on slower hardware or Debug builds
# HOW:  Set 60s timeout for tests that require extensive computation

# NOTE: Only set properties for tests that actually exist
# Timeout values set based on test complexity and computation requirements
set_tests_properties(
  unit_utils_test_kdtree
  PROPERTIES TIMEOUT 60
)

set_tests_properties(
  unit_cognitive_ethics_test_ethics_comprehensive
  PROPERTIES TIMEOUT 120
)

# Disabled test that doesn't exist
# set_tests_properties(
#   unit_cognitive_knowledge_test_knowledge_comprehensive
#   PROPERTIES TIMEOUT 120
# )

set_tests_properties(
  integration_test_brain_oscillations_integration
  PROPERTIES TIMEOUT 90
)

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

# ============================================================================
# Middleware Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/middleware")
    message(STATUS "Adding middleware unit tests...")
    add_subdirectory(unit/middleware)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/middleware")
    message(STATUS "Adding middleware integration tests...")
    add_subdirectory(integration/middleware)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/middleware")
    message(STATUS "Adding middleware regression tests...")
    add_subdirectory(regression/middleware)
endif()

# ============================================================================
# Neural Logic Tests (Core)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/logic")
    message(STATUS "Adding neural logic unit tests...")
    add_subdirectory(unit/core/logic)
endif()

# ============================================================================
# Core Directives Tests (Asimov's Laws, Golden Rule, Combinatorial Harm)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/directives")
    message(STATUS "Adding core directives unit tests...")
    add_subdirectory(unit/core/directives)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/directives")
    message(STATUS "Adding core directives integration tests...")
    add_subdirectory(integration/core/directives)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/directives")
    message(STATUS "Adding core directives regression tests...")
    add_subdirectory(regression/core/directives)
endif()

# ============================================================================
# Medulla Oblongata Tests (Phase 1.1)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/medulla")
    message(STATUS "Adding medulla unit tests...")
    add_subdirectory(unit/core/medulla)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/medulla")
    message(STATUS "Adding medulla integration tests...")
    add_subdirectory(integration/core/medulla)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/medulla")
    message(STATUS "Adding medulla regression tests...")
    add_subdirectory(regression/core/medulla)
endif()

# ============================================================================
# Cognitive Fault Tolerance Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive/fault_tolerance")
    message(STATUS "Adding cognitive fault tolerance unit tests...")
    add_subdirectory(unit/cognitive/fault_tolerance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/fault_tolerance")
    message(STATUS "Adding cognitive fault tolerance integration tests...")
    add_subdirectory(integration/cognitive/fault_tolerance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/fault_tolerance")
    message(STATUS "Adding cognitive fault tolerance regression tests...")
    add_subdirectory(regression/cognitive/fault_tolerance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive/reasoning")
    message(STATUS "Adding symbolic logic brain integration unit tests...")
    add_subdirectory(unit/cognitive/reasoning)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/reasoning")
    message(STATUS "Adding reasoning integration tests...")
    add_subdirectory(integration/cognitive/reasoning)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/reasoning")
    message(STATUS "Adding reasoning regression tests...")
    add_subdirectory(regression/cognitive/reasoning)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive/logic")
    message(STATUS "Adding cognitive logic unit tests...")
    add_subdirectory(unit/cognitive/logic)
endif()

# ============================================================================
# Cognitive Introspection Tests (Phase 1.5.4)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive/introspection")
    message(STATUS "Adding cognitive introspection unit tests...")
    add_subdirectory(unit/cognitive/introspection)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/introspection")
    message(STATUS "Adding cognitive introspection integration tests...")
    add_subdirectory(integration/cognitive/introspection)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/introspection")
    message(STATUS "Adding cognitive introspection regression tests...")
    add_subdirectory(regression/cognitive/introspection)
endif()

# ============================================================================
# Cognitive Bio-Async Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive")
    message(STATUS "Adding cognitive unit tests...")
    add_subdirectory(unit/cognitive)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/bio_async")
    message(STATUS "Adding cognitive bio-async integration tests...")
    add_subdirectory(integration/cognitive/bio_async)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/bio_async")
    message(STATUS "Adding cognitive bio-async regression tests...")
    add_subdirectory(regression/cognitive/bio_async)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive/mirror_neurons")
    message(STATUS "Adding cognitive mirror neurons unit tests...")
    add_subdirectory(unit/cognitive/mirror_neurons)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/mirror_neurons")
    message(STATUS "Adding cognitive mirror neurons integration tests...")
    add_subdirectory(integration/cognitive/mirror_neurons)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/mirror_neurons")
    message(STATUS "Adding cognitive mirror neurons regression tests...")
    add_subdirectory(regression/cognitive/mirror_neurons)
endif()

# ============================================================================
# Cognitive Immune System Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cognitive/immune")
    message(STATUS "Adding cognitive immune system unit tests...")
    add_subdirectory(unit/cognitive/immune)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/immune")
    message(STATUS "Adding cognitive immune system integration tests...")
    add_subdirectory(integration/cognitive/immune)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/immune")
    message(STATUS "Adding cognitive immune system regression tests...")
    add_subdirectory(regression/cognitive/immune)
endif()

# ============================================================================
# Enhanced Wellbeing Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/wellbeing")
    message(STATUS "Adding enhanced wellbeing integration tests...")
    add_subdirectory(integration/cognitive/wellbeing)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/wellbeing")
    message(STATUS "Adding enhanced wellbeing regression tests...")
    add_subdirectory(regression/cognitive/wellbeing)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/free_energy")
    message(STATUS "Adding FEP bridge regression tests...")
    add_subdirectory(regression/cognitive/free_energy)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/curiosity")
    message(STATUS "Adding Enhanced Curiosity regression tests...")
    add_subdirectory(regression/cognitive/curiosity)
endif()

# ============================================================================
# Networking Protocol Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/networking/protocol")
    message(STATUS "Adding networking protocol unit tests...")
    add_subdirectory(unit/networking/protocol)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/networking/protocol")
    message(STATUS "Adding networking protocol integration tests...")
    add_subdirectory(integration/networking/protocol)
endif()

# ============================================================================
# Neural Link Protocol Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/networking/nlp")
    message(STATUS "Adding Neural Link Protocol unit tests...")
    add_subdirectory(unit/networking/nlp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/networking/nlp")
    message(STATUS "Adding Neural Link Protocol integration tests...")
    add_subdirectory(integration/networking/nlp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/networking/nlp")
    message(STATUS "Adding Neural Link Protocol regression tests...")
    add_subdirectory(regression/networking/nlp)
endif()

# ============================================================================
# Networking Distributed Tests (Async API)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/networking/distributed")
    message(STATUS "Adding networking distributed unit tests...")
    add_subdirectory(unit/networking/distributed)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/networking/distributed")
    message(STATUS "Adding networking distributed integration tests...")
    add_subdirectory(integration/networking/distributed)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/networking/distributed")
    message(STATUS "Adding networking distributed regression tests...")
    add_subdirectory(regression/networking/distributed)
endif()

# ============================================================================
# Networking Immune Bridge Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/networking/immune")
    message(STATUS "Adding networking immune bridge unit tests...")
    add_subdirectory(unit/networking/immune)
endif()

# ============================================================================
# NLP Immune Bridge Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/nlp/immune")
    message(STATUS "Adding NLP immune bridge unit tests...")
    add_subdirectory(unit/nlp/immune)
endif()

# ============================================================================
# Brain Factory Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/brain/factory")
    message(STATUS "Adding brain factory unit tests...")
    add_subdirectory(unit/core/brain/factory)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/brain/factory")
    message(STATUS "Adding brain factory integration tests...")
    add_subdirectory(integration/core/brain/factory)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/brain/factory")
    message(STATUS "Adding brain factory regression tests...")
    add_subdirectory(regression/core/brain/factory)
endif()

# ============================================================================
# Core Brain Bio-Async Integration Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/brain")
    message(STATUS "Adding core brain bio-async integration tests...")
    add_subdirectory(integration/core/brain)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/brain")
    message(STATUS "Adding core brain regression tests...")
    add_subdirectory(regression/core/brain)
endif()

# ============================================================================
# Core Neuron Types Bio-Async Integration Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/neuron_types")
    message(STATUS "Adding neuron types bio-async integration tests...")
    add_subdirectory(integration/core/neuron_types)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/api")
    message(STATUS "Adding API regression tests (complex opt-in)...")
    add_subdirectory(regression/api)
endif()

# ============================================================================
# Brain Oscillations Tests (Phase C2.2: Complex Oscillation Support)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/brain/oscillations")
    message(STATUS "Adding brain oscillations unit tests (complex phasors)...")
    add_subdirectory(unit/core/brain/oscillations)
endif()

# ============================================================================
# Oscillations Sleep Bridge Tests (Phase 1.1: Sleep Integration)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/brain_oscillations/sleep")
    message(STATUS "Adding oscillations sleep bridge tests...")
    add_executable(unit_core_oscillations_sleep_bridge
        unit/core/brain_oscillations/sleep/test_oscillations_sleep_bridge.cpp
    )
    target_link_libraries(unit_core_oscillations_sleep_bridge
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_compile_features(unit_core_oscillations_sleep_bridge PRIVATE cxx_std_17)
    gtest_discover_tests(unit_core_oscillations_sleep_bridge
        PROPERTIES LABELS "unit;core;oscillations;sleep")
endif()

# ============================================================================
# Sleep Bridge Integration Tests (Phase 1.1: Sleep Integration)
# ============================================================================
# WHAT: Test all sleep bridges coordinating together
# WHY:  Verify plasticity and cognitive systems integrate correctly during sleep
# HOW:  Simulate sleep cycles with all bridges active

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/plasticity/sleep")
    message(STATUS "Adding plasticity sleep bridge integration tests...")
    add_executable(integration_plasticity_sleep_bridge
        integration/plasticity/sleep/test_plasticity_sleep_integration.cpp
    )
    target_link_libraries(integration_plasticity_sleep_bridge
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_compile_features(integration_plasticity_sleep_bridge PRIVATE cxx_std_17)
    gtest_discover_tests(integration_plasticity_sleep_bridge
        PROPERTIES LABELS "integration;plasticity;sleep")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive/sleep")
    message(STATUS "Adding cognitive sleep bridge integration tests...")
    add_executable(integration_cognitive_sleep_bridge
        integration/cognitive/sleep/test_cognitive_sleep_integration.cpp
    )
    target_link_libraries(integration_cognitive_sleep_bridge
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_compile_features(integration_cognitive_sleep_bridge PRIVATE cxx_std_17)
    gtest_discover_tests(integration_cognitive_sleep_bridge
        PROPERTIES LABELS "integration;cognitive;sleep")
endif()

# ============================================================================
# Sleep Bridge Regression Tests (Phase 1.1: Sleep Integration)
# ============================================================================
# WHAT: Exact value regression tests for sleep bridges
# WHY:  Prevent accidental changes to biologically-tuned parameters
# HOW:  Test exact values, boundary conditions, and monotonicity

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/plasticity/sleep")
    message(STATUS "Adding plasticity sleep bridge regression tests...")
    add_executable(regression_plasticity_sleep_bridge
        regression/plasticity/sleep/test_plasticity_sleep_regression.cpp
    )
    target_link_libraries(regression_plasticity_sleep_bridge
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_compile_features(regression_plasticity_sleep_bridge PRIVATE cxx_std_17)
    gtest_discover_tests(regression_plasticity_sleep_bridge
        PROPERTIES LABELS "regression;plasticity;sleep")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/cognitive/sleep")
    message(STATUS "Adding cognitive sleep bridge regression tests...")
    add_executable(regression_cognitive_sleep_bridge
        regression/cognitive/sleep/test_cognitive_sleep_regression.cpp
    )
    target_link_libraries(regression_cognitive_sleep_bridge
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_compile_features(regression_cognitive_sleep_bridge PRIVATE cxx_std_17)
    gtest_discover_tests(regression_cognitive_sleep_bridge
        PROPERTIES LABELS "regression;cognitive;sleep")
endif()

# ============================================================================
# Sleep Bridge E2E Pipeline Tests (Phase 1.1: Sleep Integration)
# ============================================================================
# WHAT: End-to-end tests for complete sleep bridge pipeline
# WHY:  Verify all 8 sleep bridges work together through realistic sleep cycles
# HOW:  Simulate full night sleep with all bridges active

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/e2e/e2e_test_sleep_bridge_pipeline.cpp")
    message(STATUS "Adding sleep bridge e2e pipeline tests...")
    add_executable(e2e_test_sleep_bridge_pipeline
        e2e/e2e_test_sleep_bridge_pipeline.cpp
    )
    target_link_libraries(e2e_test_sleep_bridge_pipeline
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_compile_features(e2e_test_sleep_bridge_pipeline PRIVATE cxx_std_17)
    gtest_discover_tests(e2e_test_sleep_bridge_pipeline
        PROPERTIES LABELS "e2e;sleep;pipeline")
endif()

# ============================================================================
# Brain Regions Tests (Broca's Area + Immune Integration)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/brain/regions")
    message(STATUS "Adding brain regions unit tests (Broca immune integration)...")
    add_subdirectory(unit/core/brain/regions)
endif()

# ============================================================================
# Hemispheric Brain Tests (Bilateral Brain Architecture)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/brain/hemispheric")
    message(STATUS "Adding hemispheric brain unit tests...")
    add_subdirectory(unit/core/brain/hemispheric)
endif()

# ============================================================================
# Utils Math Tests (Phase 0.5: Complex Number Support)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/math")
    message(STATUS "Adding utils math unit tests (complex numbers)...")
    add_subdirectory(unit/utils/math)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/math")
    message(STATUS "Adding utils math integration tests (complex oscillations)...")
    add_subdirectory(integration/utils/math)
endif()

# Utils Signal Processing Tests (Phase 0.5: Band-pass filtering for PAC)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/signal")
    message(STATUS "Adding utils signal processing unit tests...")
    add_subdirectory(unit/utils/signal)
endif()

# Utils Fault Tolerance Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/fault_tolerance")
    message(STATUS "Adding utils fault tolerance unit tests...")
    add_subdirectory(unit/utils/fault_tolerance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/fault_tolerance")
    message(STATUS "Adding utils fault tolerance integration tests...")
    add_subdirectory(integration/utils/fault_tolerance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/fault_tolerance")
    message(STATUS "Adding utils fault tolerance regression tests...")
    add_subdirectory(regression/utils/fault_tolerance)
endif()

# Utils Memory Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/memory")
    message(STATUS "Adding utils memory unit tests...")
    add_subdirectory(unit/utils/memory)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/memory")
    message(STATUS "Adding utils memory integration tests...")
    add_subdirectory(integration/utils/memory)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/memory")
    message(STATUS "Adding utils memory regression tests...")
    add_subdirectory(regression/utils/memory)
endif()

# Utils Containers Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/containers")
    message(STATUS "Adding utils containers integration tests...")
    add_subdirectory(integration/utils/containers)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/containers")
    message(STATUS "Adding utils containers regression tests...")
    add_subdirectory(regression/utils/containers)
endif()

# Utils Tensor Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/tensor")
    message(STATUS "Adding utils tensor unit tests...")
    add_subdirectory(unit/utils/tensor)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/tensor")
    message(STATUS "Adding utils tensor integration tests...")
    add_subdirectory(integration/utils/tensor)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/tensor")
    message(STATUS "Adding utils tensor regression tests...")
    add_subdirectory(regression/utils/tensor)
endif()

# Utils Encoding Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/encoding")
    message(STATUS "Adding utils encoding unit tests...")
    add_subdirectory(unit/utils/encoding)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/encoding")
    message(STATUS "Adding utils encoding integration tests (PE + Attention, WM)...")
    add_subdirectory(integration/utils/encoding)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/encoding")
    message(STATUS "Adding utils encoding regression tests...")
    add_subdirectory(regression/utils/encoding)
endif()

# ============================================================================
# Utils Gabor Filter Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/gabor")
    message(STATUS "Adding utils gabor filter unit tests...")
    add_subdirectory(unit/utils/gabor)
endif()

# ============================================================================
# Utils Ternary Logic Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/ternary")
    message(STATUS "Adding utils ternary logic unit tests...")
    add_subdirectory(unit/utils/ternary)
endif()

# ============================================================================
# Cognitive Integration Tests (General)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/cognitive")
    message(STATUS "Adding cognitive integration tests (WM + PE, emotions)...")
    add_subdirectory(integration/cognitive)
endif()

# ============================================================================
# Brain Regions Integration Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/brain_regions")
    message(STATUS "Adding brain regions integration tests (Predictive + PE)...")
    add_subdirectory(integration/core/brain_regions)
endif()

# ============================================================================
# Glial Tests (Microglia, Oligodendrocytes)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/glial")
    message(STATUS "Adding glial unit tests...")
    add_subdirectory(unit/glial)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/glial")
    message(STATUS "Adding glial integration tests...")
    add_subdirectory(integration/glial)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/glial")
    message(STATUS "Adding glial performance regression tests...")
    add_subdirectory(regression/glial)
endif()

# ============================================================================
# Axon Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/axon")
    message(STATUS "Adding axon unit tests...")
    add_subdirectory(unit/core/axon)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/axon")
    message(STATUS "Adding axon integration tests...")
    add_subdirectory(integration/core/axon)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/axon")
    message(STATUS "Adding axon performance regression tests...")
    add_subdirectory(regression/core/axon)
endif()

# ============================================================================
# Cortical Columns Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/cortical_columns")
    message(STATUS "Adding cortical columns unit tests...")
    add_subdirectory(unit/core/cortical_columns)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/cortical_columns")
    message(STATUS "Adding cortical columns integration tests...")
    add_subdirectory(integration/core/cortical_columns)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/cortical_columns")
    message(STATUS "Adding cortical columns performance regression tests...")
    add_subdirectory(regression/core/cortical_columns)
endif()

# ============================================================================
# Neuralnet Tests (Homeostasis-Immune Integration)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/neuralnet")
    message(STATUS "Adding neuralnet unit tests...")
    add_subdirectory(unit/core/neuralnet)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/neuralnet")
    message(STATUS "Adding neuralnet integration tests...")
    add_subdirectory(integration/core/neuralnet)
endif()

# ============================================================================
# Dendrite Tests (Phase 1.5.7)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/dendrite")
    message(STATUS "Adding dendrite unit tests...")
    add_subdirectory(unit/core/dendrite)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/core/dendrite")
    message(STATUS "Adding dendrite integration tests...")
    add_subdirectory(integration/core/dendrite)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/dendrite")
    message(STATUS "Adding dendrite performance regression tests...")
    add_subdirectory(regression/core/dendrite)
endif()

# ============================================================================
# Substrate Bridge Tests (Neural Substrate Integration)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/core/neural_substrate")
    message(STATUS "Adding neural substrate unit tests...")
    add_subdirectory(unit/core/neural_substrate)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/substrate")
    message(STATUS "Adding core substrate bridges regression tests...")
    add_subdirectory(regression/core/substrate)
endif()

# Neuron Types Regression Tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/core/neuron_types")
    message(STATUS "Adding neuron types performance regression tests...")
    add_subdirectory(regression/core/neuron_types)
endif()

# ============================================================================
# Security Tests (Phase SC-1: Security Coverage Framework)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/security")
    message(STATUS "Adding security unit tests...")
    add_subdirectory(unit/security)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/security")
    message(STATUS "Adding security integration tests...")
    add_subdirectory(integration/security)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/security")
    message(STATUS "Adding security regression tests...")
    add_subdirectory(regression/security)
endif()

# ============================================================================
# Utils Logging Tests (Phase LOG-1: Enhanced Logging System)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/logging")
    message(STATUS "Adding utils logging unit tests...")
    add_subdirectory(unit/utils/logging)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/logging")
    message(STATUS "Adding utils logging integration tests...")
    add_subdirectory(integration/utils/logging)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/logging")
    message(STATUS "Adding utils logging regression tests...")
    add_subdirectory(regression/utils/logging)
endif()

# ============================================================================
# Async Tests (Phase BIO-ASYNC: Biologically-Inspired Async)
# ============================================================================
# WHAT: Tests for biological async system (neuromodulators, phase coupling, etc.)
# WHY:  Verify correctness and performance of biologically-plausible async primitives
# HOW:  Unit tests verify components, integration tests verify interactions,
#       regression tests verify performance and numerical stability

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/async")
    message(STATUS "Adding async unit tests...")
    add_subdirectory(unit/async)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/async")
    message(STATUS "Adding async integration tests...")
    add_subdirectory(integration/async)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/async")
    message(STATUS "Adding async regression tests...")
    add_subdirectory(regression/async)
endif()

# ============================================================================
# Plasticity Tests
# ============================================================================
# WHAT: Tests for plasticity modules (STDP, BCM, homeostatic, dendritic, etc.)
# WHY:  Verify synaptic plasticity mechanisms and coordinator
# HOW:  Unit tests for mechanisms, coordinator tests for conflict resolution

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/plasticity")
    message(STATUS "Adding plasticity unit tests...")
    add_subdirectory(unit/plasticity)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/plasticity")
    message(STATUS "Adding plasticity integration tests...")
    add_subdirectory(integration/plasticity)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/plasticity")
    message(STATUS "Adding plasticity regression tests...")
    add_subdirectory(regression/plasticity)
endif()

# ============================================================================
# Perception Tests
# ============================================================================
# WHAT: Tests for perception modules (visual, audio, speech cortex)
# WHY:  Verify sensory processing and cortical integration
# HOW:  Unit tests for cortical processing, integration tests for multimodal

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/perception")
    message(STATUS "Adding perception unit tests...")
    add_subdirectory(unit/perception)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/perception")
    message(STATUS "Adding perception integration tests...")
    add_subdirectory(integration/perception)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/perception")
    message(STATUS "Adding perception regression tests...")
    add_subdirectory(regression/perception)
endif()

# ============================================================================
# Swarm Intelligence Tests (Phase SWARM-1: Drone Collective Intelligence)
# ============================================================================
# WHAT: Tests for swarm brain coordination system
# WHY:  Verify distributed cognition, consensus, and emergence detection
# HOW:  Unit tests for components, integration tests for multi-drone coordination

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/swarm")
    message(STATUS "Adding swarm unit tests...")
    add_subdirectory(unit/swarm)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/swarm")
    message(STATUS "Adding swarm integration tests...")
    add_subdirectory(integration/swarm)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/swarm")
    message(STATUS "Adding swarm regression tests...")
    add_subdirectory(regression/swarm)
endif()

# ============================================================================
# Platform Tier Tests (Portia Spider Foundation)
# ============================================================================
# WHAT: Tests for platform tier detection and configuration
# WHY:  Verify correct resource adaptation across hardware tiers
# HOW:  Unit tests for tier detection, integration tests for brain configuration

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/platform")
    message(STATUS "Adding platform tier unit tests...")
    add_subdirectory(unit/utils/platform)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/utils/platform")
    message(STATUS "Adding platform tier integration tests...")
    add_subdirectory(integration/utils/platform)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/utils/platform")
    message(STATUS "Adding platform tier regression tests...")
    add_subdirectory(regression/utils/platform)
endif()

# ============================================================================
# Sparse Synapse Tests (Memory Optimization)
# ============================================================================
# Already added above in neuralnet section - removed duplicate

# ============================================================================
# Portia Spider Tests (Graceful Degradation, Learning, Sensor Fusion)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/portia")
    message(STATUS "Adding portia spider unit tests...")
    add_subdirectory(unit/portia)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/portia")
    message(STATUS "Adding portia spider integration tests...")
    add_subdirectory(integration/portia)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/portia")
    message(STATUS "Adding portia spider regression tests...")
    add_subdirectory(regression/portia)
endif()

# ============================================================================
# GPU Immune Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu/immune/CMakeLists.txt")
    message(STATUS "Adding GPU immune unit tests...")
    add_subdirectory(unit/gpu/immune)
endif()

# ============================================================================
# LNN Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/lnn/CMakeLists.txt")
    message(STATUS "Adding LNN unit tests...")
    add_subdirectory(unit/lnn)
endif()

# ============================================================================
# SNN Tests (Spiking Neural Network)
# ============================================================================
# WHAT: Tests for SNN module integrating with existing neural infrastructure
# WHY:  SNN is the core neural substrate providing spike-based computation
# HOW:  Tests config, network, encoding, training, and bio-async integration

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/snn/CMakeLists.txt")
    message(STATUS "Adding SNN unit tests...")
    add_subdirectory(unit/snn)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/snn/CMakeLists.txt")
    message(STATUS "Adding SNN integration tests...")
    add_subdirectory(integration/snn)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/snn/CMakeLists.txt")
    message(STATUS "Adding SNN regression tests...")
    add_subdirectory(regression/snn)
endif()

# ============================================================================
# Performance Benchmarks
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    message(STATUS "Adding performance benchmarks...")
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Utils Quantum Tests (Quantum Walk for Graph Algorithms)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/quantum")
    message(STATUS "Adding utils quantum unit tests...")
    add_subdirectory(unit/utils/quantum)
endif()

# ============================================================================
# Quantum Bridge Integration Tests
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/quantum")
    message(STATUS "Adding quantum bridge unit tests...")
    add_subdirectory(unit/quantum)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/quantum")
    message(STATUS "Adding quantum bridge integration tests...")
    add_subdirectory(integration/quantum)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression/quantum")
    message(STATUS "Adding quantum bridge regression tests...")
    add_subdirectory(regression/quantum)
endif()

