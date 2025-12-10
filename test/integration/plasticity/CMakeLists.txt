# Integration tests for plasticity bio-async interactions

add_executable(test_plasticity_bioasync_integration
    test_plasticity_bioasync_integration.cpp
)

target_link_libraries(test_plasticity_bioasync_integration
    nimcp_unified_memory
    nimcp_bio_async
    nimcp_bio_router
    nimcp_stdp
    nimcp_homeostatic
    nimcp_attention
    nimcp_dendritic
    nimcp_adaptive
    nimcp_predictive_coding
    nimcp_neuromodulators
    nimcp_receptor_subtypes
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME PlasticityBioAsyncIntegration
    COMMAND test_plasticity_bioasync_integration
)

# Mark as integration test (slower)
set_tests_properties(PlasticityBioAsyncIntegration PROPERTIES
    TIMEOUT 300
    LABELS "integration;plasticity;bioasync"
)

# Second messenger cascade integration tests
add_executable(integration_plasticity_second_messengers_test
    second_messengers/test_second_messengers_integration.cpp
)
target_link_libraries(integration_plasticity_second_messengers_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(integration_plasticity_second_messengers_test PRIVATE cxx_std_17)
gtest_discover_tests(integration_plasticity_second_messengers_test
    PROPERTIES
        TIMEOUT 300
        LABELS "integration;plasticity;second_messengers"
)

# Second messenger module integration tests (Broca, Audio, Speech, etc.)
add_executable(integration_plasticity_second_messengers_module_test
    second_messengers/test_second_messengers_module_integration.cpp
)
target_link_libraries(integration_plasticity_second_messengers_module_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(integration_plasticity_second_messengers_module_test PRIVATE cxx_std_17)
gtest_discover_tests(integration_plasticity_second_messengers_module_test
    PROPERTIES
        TIMEOUT 300
        LABELS "integration;plasticity;second_messengers;modules"
)
