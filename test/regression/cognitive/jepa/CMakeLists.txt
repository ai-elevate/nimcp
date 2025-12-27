# ============================================================================
# JEPA Regression Tests
# ============================================================================
# WHAT: Build configuration for JEPA module regression tests
# WHY:  Verify stability, determinism, and performance of JEPA components
# HOW:  Single executable with all regression tests for comprehensive validation

# JEPA Regression Test Executable
add_executable(regression_jepa
    test_jepa_regression.cpp
)

target_link_libraries(regression_jepa
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_jepa
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_jepa PRIVATE cxx_std_17)
target_compile_definitions(regression_jepa PRIVATE NIMCP_TESTING)

# Register tests with CTest
gtest_discover_tests(regression_jepa
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;jepa"
        TIMEOUT 600
)
