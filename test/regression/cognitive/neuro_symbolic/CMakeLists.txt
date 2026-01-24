# ============================================================================
# Neuro-Symbolic Mathematics Regression Tests
# ============================================================================
# WHAT: Regression tests for Mathesis-inspired genius-level math system
# WHY:  Ensure API contracts, edge cases, and boundary conditions remain stable
# HOW:  Test edge cases, boundary conditions, memory safety, and API contracts
# ============================================================================

add_executable(regression_neuro_symbolic
    test_neuro_symbolic_regression.cpp
)

target_link_libraries(regression_neuro_symbolic
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_neuro_symbolic
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(regression_neuro_symbolic PRIVATE cxx_std_17)

gtest_discover_tests(regression_neuro_symbolic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;neuro_symbolic"
        TIMEOUT 120
)

message(STATUS "  Added neuro-symbolic mathematics regression tests")
