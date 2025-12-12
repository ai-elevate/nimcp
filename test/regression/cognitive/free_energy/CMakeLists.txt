# ============================================================================
# Free Energy Principle Regression Tests
# ============================================================================
# WHAT: Build configuration for FEP module regression tests
# WHY:  Verify stability, performance, and numerical accuracy of FEP modules
# HOW:  Single executable with all regression tests for long-running validation

add_executable(regression_fep_enhanced
    test_fep_learning_regression.cpp
    test_fep_curiosity_regression.cpp
    test_fep_neuromod_regression.cpp
    test_fep_immune_regression.cpp
    test_fep_consciousness_regression.cpp
    test_fep_planning_regression.cpp
    test_fep_evidence_regression.cpp
    test_fep_context_regression.cpp
    test_fep_sleep_regression.cpp
)
target_link_libraries(regression_fep_enhanced PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)
target_include_directories(regression_fep_enhanced PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(regression_fep_enhanced PRIVATE cxx_std_17)
target_compile_definitions(regression_fep_enhanced PRIVATE NIMCP_TESTING)
gtest_discover_tests(regression_fep_enhanced
    PROPERTIES LABELS "regression;cognitive;free_energy"
    TIMEOUT 300
)
