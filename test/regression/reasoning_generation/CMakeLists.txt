# =============================================================================
# NIMCP Reasoning-Generation Regression Tests
# =============================================================================
# WHAT: Regression tests guarding against specific edge-case bugs
# WHY:  Prevent regressions in NULL handling, buffer overflow, NaN propagation,
#       and encode/decode round-trip correctness
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(regression_reasoning_generation_test
    test_reasoning_generation_regression.cpp
)

target_link_libraries(regression_reasoning_generation_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_reasoning_generation_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_reasoning_generation_test PRIVATE cxx_std_17)

gtest_discover_tests(regression_reasoning_generation_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;reasoning;generation"
        TIMEOUT 300
)

# =============================================================================
# Symbolic Logic Reasoning Regression Tests
# =============================================================================
# WHAT: Regression tests for symbolic logic integration in the reasoning chain
# WHY:  Prevent regressions in NULL brain handling, empty KB, extreme config
#       values, and long query buffer overflow
# HOW:  GTest executable linked against nimcp library
# =============================================================================

add_executable(regression_symbolic_logic_reasoning_test
    test_symbolic_logic_reasoning_regression.cpp
)

target_link_libraries(regression_symbolic_logic_reasoning_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_symbolic_logic_reasoning_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(regression_symbolic_logic_reasoning_test PRIVATE cxx_std_17)

gtest_discover_tests(regression_symbolic_logic_reasoning_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;reasoning;symbolic-logic"
        TIMEOUT 300
)
