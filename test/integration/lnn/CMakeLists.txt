# ============================================================================
# LNN Integration Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for LNN integration tests
# WHY:  Verify LNN integration with other modules (substrate, etc.)
# HOW:  Build GoogleTest executables linking required modules
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping LNN integration tests")
    return()
endif()

# Common link libraries for LNN integration tests
set(LNN_INTEGRATION_TEST_LIBS
    nimcp_lnn
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)

# Common include directories
set(LNN_INTEGRATION_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/lnn
)

# ============================================================================
# LNN Substrate Integration Test (2026-04-22)
# ============================================================================
# WHAT: Integration tests for the LNN substrate adapter (Phase 2 biological-
#       substrate wiring) running inside a multi-step forward loop.
# WHY:  The unit tests isolate the effects cache + tunables; integration is
#       where composition with the learned LNN dynamics is verified over
#       many steps. The adapter must observably modulate outputs when "on"
#       and be a bit-perfect no-op when "off".
# HOW:  Google Test. Small 1-layer LNN, CPU-only forward, substrate
#       ramp-down over 100 steps vs. no-substrate baseline.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_lnn_substrate_integration.cpp)
    add_executable(integration_lnn_test_lnn_substrate_integration
        test_lnn_substrate_integration.cpp
    )

    target_link_libraries(integration_lnn_test_lnn_substrate_integration
        PRIVATE ${LNN_INTEGRATION_TEST_LIBS}
    )

    target_include_directories(integration_lnn_test_lnn_substrate_integration
        PRIVATE ${LNN_INTEGRATION_TEST_INCLUDES}
    )

    target_compile_features(integration_lnn_test_lnn_substrate_integration
        PRIVATE cxx_std_17)

    gtest_discover_tests(integration_lnn_test_lnn_substrate_integration
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;lnn;substrate"
            TIMEOUT 300
    )

    message(STATUS "  Added LNN substrate integration test: "
                   "test_lnn_substrate_integration")
endif()
