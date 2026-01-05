# ============================================================================
# Language Layer Integration Tests
# ============================================================================
# WHAT: Integration tests for Language Layer
# WHY:  Verify end-to-end language processing flows and bridge coordination
# HOW:  gtest-based tests for full pipelines and multi-bridge scenarios
# ============================================================================

# ============================================================================
# Language Integration Tests
# ============================================================================
add_executable(integration_language
    test_language_integration.cpp
)
target_link_libraries(integration_language
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(integration_language PRIVATE cxx_std_17)
gtest_discover_tests(integration_language
    PROPERTIES LABELS "integration;language")

