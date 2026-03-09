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

# ============================================================================
# SNN Bridge Integration Test
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_bridge_integration.cpp)
    add_executable(integration_snn_test_bridge_integration
        test_snn_bridge_integration.cpp
    )

    target_link_libraries(integration_snn_test_bridge_integration
        PRIVATE ${SNN_INTEGRATION_TEST_LIBS}
    )

    target_include_directories(integration_snn_test_bridge_integration
        PRIVATE ${SNN_INTEGRATION_TEST_INCLUDES}
    )

    target_compile_features(integration_snn_test_bridge_integration PRIVATE cxx_std_17)

    gtest_discover_tests(integration_snn_test_bridge_integration
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;snn;bridges"
            TIMEOUT 120
    )

    message(STATUS "  Added SNN bridge integration test: test_snn_bridge_integration")
endif()

# ============================================================================
# SNN-Plasticity Integration Test
# ============================================================================
# WHAT: Test STDP and BCM plasticity with full SNN network training
# WHY:  Verify eligibility traces and threshold dynamics
# HOW:  Test plasticity bridges with reward signals

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_plasticity_integration.cpp)
    add_executable(integration_snn_plasticity
        test_snn_plasticity_integration.cpp
    )

    target_link_libraries(integration_snn_plasticity
        PRIVATE ${SNN_INTEGRATION_TEST_LIBS}
    )

    target_include_directories(integration_snn_plasticity
        PRIVATE ${SNN_INTEGRATION_TEST_INCLUDES}
    )

    target_compile_features(integration_snn_plasticity PRIVATE cxx_std_17)

    gtest_discover_tests(integration_snn_plasticity
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;snn;plasticity;stdp;bcm"
            TIMEOUT 180
    )

    message(STATUS "  Added SNN plasticity integration test: test_snn_plasticity_integration")
endif()

# ============================================================================
# SNN Language Bridge Integration Tests
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_language_bridge_integration.cpp)
    add_executable(integration_snn_language_bridge
        test_snn_language_bridge_integration.cpp
    )

    target_link_libraries(integration_snn_language_bridge
        PRIVATE ${SNN_INTEGRATION_TEST_LIBS}
    )

    target_include_directories(integration_snn_language_bridge
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    target_compile_features(integration_snn_language_bridge PRIVATE cxx_std_17)

    gtest_discover_tests(integration_snn_language_bridge
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;snn;language;bridge"
            TIMEOUT 180
    )

    message(STATUS "  Added SNN language bridge integration test: test_snn_language_bridge_integration")
endif()

# ============================================================================
# Include Subdirectories
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exception)
    add_subdirectory(exception)
endif()
