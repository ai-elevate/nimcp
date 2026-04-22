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
# SNN Biophysical Integration Test (2026-04-22)
# ============================================================================
# WHAT: Live integration tests for the five biophysical stability mechanisms
#       (AHP, Na+/K+ pump, basket-cell inhibition, E/I-balanced noise,
#       anti-reward) exercised through snn_network_step and snn_rstdp_apply.
# WHY:  Unit tests cover each mechanism in isolation; this test verifies each
#       knob observably changes pop-level firing or synaptic dynamics in the
#       actual runtime hot path when toggled on/off.
# HOW:  Google Test. Creates lightweight (CSR) populations, forces CPU
#       fallback so the biophysics path runs, and compares baseline vs
#       feature-enabled spike counts / peak rates / synaptic weight deltas.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_biophysical_integration.cpp)
    add_executable(integration_snn_test_snn_biophysical_integration
        test_snn_biophysical_integration.cpp
    )

    target_link_libraries(integration_snn_test_snn_biophysical_integration
        PRIVATE
            nimcp_snn
            nimcp
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            m
            z
    )

    target_include_directories(integration_snn_test_snn_biophysical_integration
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include/snn
    )

    target_compile_features(integration_snn_test_snn_biophysical_integration
        PRIVATE cxx_std_17)

    gtest_discover_tests(integration_snn_test_snn_biophysical_integration
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;snn;biophysical"
            TIMEOUT 300
    )

    message(STATUS "  Added SNN biophysical integration test: "
                   "test_snn_biophysical_integration")
endif()

# ============================================================================
# SNN Thalamic Substrate Composition Test (2026-04-22)
# ============================================================================
# WHAT: Integration test verifying the existing SNN thalamic bridge composes
#       correctly with Wave A+B biophysical features (basket pool, AHP,
#       Na+/K+ pump, E/I-balanced noise, anti-reward).
# WHY:  The bridge predates these features. Wave A+B changes the distribution
#       of pop->spike_output — prove the bridge still routes, attention
#       gates stay finite, and burst-mode thresholds remain sensible.
# HOW:  Google Test. Creates lightweight CSR populations, forces CPU
#       fallback so biophysics runs, builds the thalamic router + SNN
#       thalamic bridge on top, routes spikes through the bridge for each
#       Wave A+B knob configuration.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_thalamic_substrate_compose.cpp)
    add_executable(integration_snn_test_snn_thalamic_substrate_compose
        test_snn_thalamic_substrate_compose.cpp
    )

    target_link_libraries(integration_snn_test_snn_thalamic_substrate_compose
        PRIVATE
            nimcp_snn
            nimcp
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            m
            z
    )

    target_include_directories(integration_snn_test_snn_thalamic_substrate_compose
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include/snn
    )

    target_compile_features(integration_snn_test_snn_thalamic_substrate_compose
        PRIVATE cxx_std_17)

    gtest_discover_tests(integration_snn_test_snn_thalamic_substrate_compose
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "integration;snn;thalamic;biophysical"
            TIMEOUT 300
    )

    message(STATUS "  Added SNN thalamic substrate composition test: "
                   "test_snn_thalamic_substrate_compose")
endif()

# ============================================================================
# Include Subdirectories
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exception)
    add_subdirectory(exception)
endif()
