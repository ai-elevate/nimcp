# ============================================================================
# Parietal Lobe FEP Orchestrator Bridge Regression Tests
# ============================================================================
# WHAT: Build configuration for parietal-FEP orchestrator bridge regression tests
# WHY:  Verify stability, performance, and numerical accuracy of parietal-FEP orchestrator integration
# HOW:  Single executable with all regression tests for long-running validation
#
# NOTE: This tests the parietal_fep_bridge (FEP orchestrator integration)
#       The older fep_parietal_bridge tests are in test/regression/parietal/

# Parietal FEP Orchestrator Bridge Regression Tests
add_executable(regression_parietal_fep_orchestrator_bridge
    test_parietal_fep_regression.cpp
)

target_link_libraries(regression_parietal_fep_orchestrator_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_parietal_fep_orchestrator_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_parietal_fep_orchestrator_bridge PRIVATE cxx_std_17)
target_compile_definitions(regression_parietal_fep_orchestrator_bridge PRIVATE NIMCP_TESTING)

gtest_discover_tests(regression_parietal_fep_orchestrator_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;parietal;fep;orchestrator;bridge"
        TIMEOUT 600
)
