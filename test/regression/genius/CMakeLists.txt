# =============================================================================
# Genius Profiles Regression Tests
# =============================================================================
# WHAT: Regression tests for genius profile system stability and correctness
# WHY:  Ensure genius profiles maintain consistent behavior across versions
# HOW:  Test parameter consistency, eidetic presets, blending, error codes
# =============================================================================

# =============================================================================
# Genius Profiles Regression Test
# Tests:
# - Profile parameter consistency (all 9 types)
# - Eidetic preset reproducibility (Tesla, Mozart, von Neumann, Kim Peek, Wiltshire)
# - Blending determinism and transitivity
# - State machine transitions (UNINITIALIZED -> READY -> ACTIVE -> SUSPENDED)
# - Error code consistency
# - Bridge lifecycle (create/destroy with bridge base pattern)
# =============================================================================

add_executable(regression_genius_profiles
    test_genius_regression.cpp
)

target_link_libraries(regression_genius_profiles
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(regression_genius_profiles PRIVATE cxx_std_17)

add_test(
    NAME regression_genius_profiles
    COMMAND regression_genius_profiles
)

set_tests_properties(regression_genius_profiles
    PROPERTIES
    LABELS "regression;genius;profiles;brain"
    TIMEOUT 120
)

# =============================================================================
# Custom Target: Run All Genius Regression Tests
# =============================================================================
add_custom_target(test_genius_regression
    COMMAND ${CMAKE_CTEST_COMMAND} -L "regression" -L "genius" --output-on-failure
    DEPENDS
        regression_genius_profiles
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all genius profiles regression tests"
)
