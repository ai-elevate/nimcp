# ============================================================================
# NIMCP Utils End-to-End Tests
# ============================================================================
# WHAT: E2E tests for utility modules
# WHY:  Validate complete workflows for memory, fault tolerance, health monitoring
# HOW:  GTest-based tests with pipeline tracking
# ============================================================================

message(STATUS "  Adding Utils E2E tests...")

# NOTE: fuzzy subdirectory is included separately via test/CMakeLists.txt

# ============================================================================
# Memory + Health Agent Integration E2E Tests (P1-P3 Remediation)
# ============================================================================
# WHAT: End-to-end tests for memory operations with health agent monitoring
# WHY:  Verify health agent correctly monitors memory operations and reports issues

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_health_e2e.cpp)
    add_executable(e2e_utils_test_memory_health_e2e
        ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_health_e2e.cpp
        ${CMAKE_SOURCE_DIR}/test/e2e/e2e_test_framework.cpp
    )
    target_include_directories(e2e_utils_test_memory_health_e2e PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/utils
        ${CMAKE_SOURCE_DIR}/include/utils/fault_tolerance
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/test/e2e
    )
    target_link_libraries(e2e_utils_test_memory_health_e2e PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
    )
    target_compile_features(e2e_utils_test_memory_health_e2e PRIVATE cxx_std_17)
    target_compile_definitions(e2e_utils_test_memory_health_e2e PRIVATE
        NIMCP_TESTING
        NIMCP_E2E_TESTING
    )
    add_test(NAME e2e_utils_test_memory_health_e2e
             COMMAND e2e_utils_test_memory_health_e2e)
    set_tests_properties(e2e_utils_test_memory_health_e2e PROPERTIES
        LABELS "e2e;utils;memory;health_agent"
        TIMEOUT 600
    )
    message(STATUS "    Added: e2e_utils_test_memory_health_e2e")
endif()

message(STATUS "Added P1-P3 Remediation Utils E2E tests:")
message(STATUS "  - e2e_utils_test_memory_health_e2e (Memory + Health Agent Integration)")
