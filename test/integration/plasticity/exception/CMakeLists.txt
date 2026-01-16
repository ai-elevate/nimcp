# =============================================================================
# Integration Tests for Plasticity Exception Handling
# =============================================================================
# WHAT: CMakeLists.txt for plasticity exception integration tests
# WHY:  Test end-to-end exception flow with immune system integration
# HOW:  Build test executable and register with CTest
#
# INTEGRATION SCENARIOS:
# - Plasticity error -> Exception -> Handler -> Immune presentation
# - Batch exception processing
# - Cross-module exception propagation
# - Concurrent exception handling
# - Recovery strategy execution
#
# @author NIMCP Development Team
# @date 2026-01-16
# =============================================================================

# Plasticity Exception Integration Tests
add_executable(integration_plasticity_exception
    test_plasticity_exception_integration.cpp
)

target_include_directories(integration_plasticity_exception
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(integration_plasticity_exception
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(integration_plasticity_exception PRIVATE cxx_std_17)

# Register with CTest using gtest_discover_tests for automatic test discovery
gtest_discover_tests(integration_plasticity_exception
    PROPERTIES
        TIMEOUT 300
        LABELS "integration;plasticity;exception;immune"
)

# Alternative manual test registration (if gtest_discover_tests has issues)
# add_test(
#     NAME integration_plasticity_exception
#     COMMAND integration_plasticity_exception --gtest_brief=1
# )
#
# set_tests_properties(integration_plasticity_exception PROPERTIES
#     TIMEOUT 300
#     LABELS "integration;plasticity;exception;immune"
# )
