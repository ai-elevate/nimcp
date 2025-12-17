# ============================================================================
# Medulla Oblongata Regression Tests
# ============================================================================
#
# WHAT: Regression tests for Medulla module stability
# WHY:  Prevent regressions in arousal, protection, circadian behavior
# HOW:  Test known edge cases, boundary conditions, and historical bugs
#

# Test: Medulla Regression (boundary conditions, state consistency)
add_executable(regression_core_medulla
    regression_core_medulla.cpp
)

target_link_libraries(regression_core_medulla
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)

target_include_directories(regression_core_medulla
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_core_medulla PRIVATE cxx_std_17)

# Register test with CTest
add_test(NAME regression_core_medulla
         COMMAND regression_core_medulla --gtest_brief=1)

set_tests_properties(regression_core_medulla
    PROPERTIES
        TIMEOUT 180
        LABELS "regression;core;medulla;brainstem;stability"
)
