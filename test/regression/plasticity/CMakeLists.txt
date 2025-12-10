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

# Second messenger cascade regression tests
add_executable(regression_plasticity_second_messengers_test
    second_messengers/test_second_messengers_regression.cpp
)
target_link_libraries(regression_plasticity_second_messengers_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(regression_plasticity_second_messengers_test PRIVATE cxx_std_17)
gtest_discover_tests(regression_plasticity_second_messengers_test
    PROPERTIES
        TIMEOUT 600
        LABELS "regression;plasticity;second_messengers"
)

# Second messenger module integration regression tests
add_executable(regression_plasticity_second_messengers_module_test
    second_messengers/test_second_messengers_module_regression.cpp
)
target_link_libraries(regression_plasticity_second_messengers_module_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(regression_plasticity_second_messengers_module_test PRIVATE cxx_std_17)
gtest_discover_tests(regression_plasticity_second_messengers_module_test
    PROPERTIES
        TIMEOUT 600
        LABELS "regression;plasticity;second_messengers;modules"
)

# Positional encoding regression tests for attention
add_executable(regression_plasticity_attention_pe_test
    attention/test_attention_pe_regression.cpp
)
target_link_libraries(regression_plasticity_attention_pe_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(regression_plasticity_attention_pe_test PRIVATE cxx_std_17)
gtest_discover_tests(regression_plasticity_attention_pe_test
    PROPERTIES
        TIMEOUT 600
        LABELS "regression;plasticity;attention;positional_encoding"
)
