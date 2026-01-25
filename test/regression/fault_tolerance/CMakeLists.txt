# ============================================================================
# Fault Tolerance Regression Tests
# ============================================================================
# WHAT: Regression tests for health system stability
# WHY:  Phase 8 health integration requires stability testing
# HOW:  GoogleTest-based tests for edge cases and boundary conditions
# ============================================================================

# Main Fault Tolerance Regression Tests
add_executable(regression_fault_tolerance
    test_fault_tolerance_regression.cpp
)

target_include_directories(regression_fault_tolerance PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_fault_tolerance
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_fault_tolerance PRIVATE cxx_std_17)
target_compile_definitions(regression_fault_tolerance PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_fault_tolerance
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance"
        TIMEOUT 180
)

# Health Stability Regression Tests (Phase 8)
add_executable(regression_health_stability
    test_health_stability_regression.cpp
)

target_include_directories(regression_health_stability PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(regression_health_stability
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(regression_health_stability PRIVATE cxx_std_17)
target_compile_definitions(regression_health_stability PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_health_stability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;fault_tolerance;health;phase8"
        TIMEOUT 180
)

message(STATUS "Added fault tolerance regression tests")
