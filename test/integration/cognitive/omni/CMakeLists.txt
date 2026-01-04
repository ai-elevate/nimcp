# Omnidirectional Inference Integration Tests

# TODO: Integration test needs API updates to match actual implementation:
# - jepa_bidirectional_predict now takes jepa_latent_t* not float*
# - predictive_hierarchy_forward -> pred_hier_forward_pass
# - temporal_replay_store_sequence -> temporal_replay_store_transition
# - Many other API signature changes
if(GTest_FOUND AND FALSE)
    # Test executable for omnidirectional bridge integration
    add_executable(test_omni_bridges_integration
        test_omni_bridges_integration.cpp
    )

    target_link_libraries(test_omni_bridges_integration
        nimcp
        GTest::gtest
        GTest::gtest_main
    )

    target_include_directories(test_omni_bridges_integration PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    # Register with CTest
    add_test(NAME OmniBridgesIntegration
        COMMAND test_omni_bridges_integration
    )

    # Set test properties
    set_tests_properties(OmniBridgesIntegration PROPERTIES
        TIMEOUT 120
        LABELS "integration;cognitive;omni"
    )
endif()
