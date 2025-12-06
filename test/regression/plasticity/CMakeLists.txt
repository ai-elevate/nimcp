# Regression tests for plasticity learning stability

add_executable(test_learning_stability
    test_learning_stability.cpp
)

target_link_libraries(test_learning_stability
    nimcp_unified_memory
    nimcp_bio_async
    nimcp_bio_router
    nimcp_stdp
    nimcp_homeostatic
    nimcp_bcm
    nimcp_predictive_coding
    nimcp_adaptive
    nimcp_neuromodulators
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME LearningStabilityRegression
    COMMAND test_learning_stability
)

# Mark as regression test (slowest, most thorough)
set_tests_properties(LearningStabilityRegression PROPERTIES
    TIMEOUT 600
    LABELS "regression;plasticity;stability"
)
