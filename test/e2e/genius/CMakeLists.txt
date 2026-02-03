# =============================================================================
# Genius Profiles End-to-End Tests
# =============================================================================
# WHAT: End-to-end workflow tests for genius profile system
# WHY:  Validate complete genius profile lifecycle in realistic scenarios
# HOW:  Test full workflows including creation, training, cognitive sessions
# =============================================================================

# =============================================================================
# Genius Profiles E2E Test
# Tests:
# - Complete profile lifecycle (create -> configure -> activate -> destroy)
# - Polymath creation workflow (combine multiple genius types)
# - Cognitive session simulation (stimulus -> processing -> response)
# - Immune system modulation during high-load scenarios
# - Training pipeline integration
# - Multi-profile concurrent sessions
# - Stress tests (high iteration counts, memory pressure)
# - Full workday simulation (8-hour cognitive workload)
# =============================================================================

add_executable(e2e_genius_profiles
    test_genius_e2e.cpp
)

target_link_libraries(e2e_genius_profiles
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(e2e_genius_profiles PRIVATE cxx_std_17)

add_test(
    NAME e2e_genius_profiles
    COMMAND e2e_genius_profiles
)

set_tests_properties(e2e_genius_profiles
    PROPERTIES
    LABELS "e2e;genius;profiles;brain;workflow"
    TIMEOUT 300
)

# =============================================================================
# Custom Target: Run All Genius E2E Tests
# =============================================================================
add_custom_target(test_genius_e2e
    COMMAND ${CMAKE_CTEST_COMMAND} -L "e2e" -L "genius" --output-on-failure
    DEPENDS
        e2e_genius_profiles
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all genius profiles E2E tests"
)
