# ============================================================================
# Language Layer Unit Tests
# ============================================================================
# WHAT: Unit tests for Language Layer
# WHY:  Verify language orchestrator and bridges functionality
# HOW:  gtest-based tests for lifecycle, config, and operations
# ============================================================================

# ============================================================================
# Language Orchestrator Tests
# ============================================================================
add_executable(unit_language_orchestrator
    test_language_orchestrator.cpp
)
target_link_libraries(unit_language_orchestrator
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_language_orchestrator PRIVATE cxx_std_17)
gtest_discover_tests(unit_language_orchestrator
    PROPERTIES LABELS "unit;language;orchestrator")

# ============================================================================
# Language Bridges Tests
# ============================================================================
add_executable(unit_language_bridges
    test_language_bridges.cpp
)
target_link_libraries(unit_language_bridges
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_language_bridges PRIVATE cxx_std_17)
gtest_discover_tests(unit_language_bridges
    PROPERTIES LABELS "unit;language;bridges")
