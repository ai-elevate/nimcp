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
