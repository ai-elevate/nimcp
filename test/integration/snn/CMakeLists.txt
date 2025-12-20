# ============================================================================
# SNN Integration Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for SNN integration tests
# WHY:  Verify SNN integration with other modules
# HOW:  Build GoogleTest executable linking all required modules
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping SNN integration tests")
    return()
endif()

# Common link libraries for SNN integration tests
set(SNN_INTEGRATION_TEST_LIBS
    nimcp_snn
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)

# Common include directories
set(SNN_INTEGRATION_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/snn
)

# ============================================================================
# SNN Integration Test
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_integration.cpp)
    add_executable(integration_snn_test_integration
        test_snn_integration.cpp
    )

    target_link_libraries(integration_snn_test_integration
        PRIVATE ${SNN_INTEGRATION_TEST_LIBS}
    )

    target_include_directories(integration_snn_test_integration
        PRIVATE ${SNN_INTEGRATION_TEST_INCLUDES}
    )

    target_compile_features(integration_snn_test_integration PRIVATE cxx_std_17)

    gtest_discover_tests(integration_snn_test_integration
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;snn"
            TIMEOUT 120
    )

    message(STATUS "  Added SNN integration test: test_snn_integration")
endif()
