# Middleware Middleware Integration Tests
# Test Middleware Subsystems Integration
add_executable(integration_middleware_middleware_test_middleware_subsystems_integration
    test_middleware_subsystems_integration.cpp
)
target_link_libraries(integration_middleware_middleware_test_middleware_subsystems_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME integration_middleware_middleware_test_middleware_subsystems_integration
         COMMAND integration_middleware_middleware_test_middleware_subsystems_integration)

# Pipeline Integration Tests
add_executable(integration_middleware_pipeline_integration
    pipeline/test_pipeline_integration.cpp
)
target_link_libraries(integration_middleware_pipeline_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME integration_middleware_pipeline_integration
         COMMAND integration_middleware_pipeline_integration)

# Phase 2 Integration Tests
# TODO: Fix outdated API usage in test_phase2_integration.cpp
# - Uses old spike_record_t instead of spike_train_t
# - Uses old network_config_t and feature_extractor API
# Temporarily disabled until Phase 3 implementation is complete
#add_executable(integration_middleware_middleware_test_phase2_integration
#    test_phase2_integration.cpp
#)
#target_link_libraries(integration_middleware_middleware_test_phase2_integration
#    PRIVATE
#        GTest::GTest
#        GTest::Main
#        nimcp_middleware
#        nimcp
#    )
#add_test(NAME integration_middleware_middleware_test_phase2_integration
#         COMMAND integration_middleware_middleware_test_phase2_integration)
