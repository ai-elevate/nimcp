# Utils Fault Tolerance Integration Tests
# Enhanced Fault Tolerance Module Integration (Phase FT-2: Immune System)

# Helper macro to add integration test
macro(add_ft_integration_test TEST_NAME)
    add_executable(integration_utils_fault_tolerance_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(integration_utils_fault_tolerance_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            pthread
    )
    target_include_directories(integration_utils_fault_tolerance_${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME integration_utils_fault_tolerance_${TEST_NAME}
             COMMAND integration_utils_fault_tolerance_${TEST_NAME})
    set_tests_properties(integration_utils_fault_tolerance_${TEST_NAME} PROPERTIES LABELS "integration")
endmacro()

# Existing integration tests
add_ft_integration_test(test_async_checkpoint_integration)
add_ft_integration_test(test_brain_recovery_integration)
add_ft_integration_test(test_checkpoint_recovery)
add_ft_integration_test(test_diagnostics_integration)
add_ft_integration_test(test_fast_recovery_integration)
add_ft_integration_test(test_fault_event_bus_integration)
add_ft_integration_test(test_fault_tolerance_integration)
add_ft_integration_test(test_lockfree_metrics_integration)
add_ft_integration_test(test_recovery_cache_integration)
add_ft_integration_test(test_recovery_pool_integration)

# Enhanced Fault Tolerance Integration Tests (Phase FT-2)
add_ft_integration_test(test_enhanced_fault_tolerance_integration)

# Health Agent USE Function Integration Tests (Phase 1)
add_ft_integration_test(test_health_agent_use_integration)
set_tests_properties(integration_utils_fault_tolerance_test_health_agent_use_integration
                     PROPERTIES TIMEOUT 180)
