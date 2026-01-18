# Fault Tolerance Unit Tests
# All tests in this directory are explicitly managed here

# Helper macro to add fault tolerance test
macro(add_fault_tolerance_test TEST_NAME)
    add_executable(unit_utils_fault_tolerance_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_utils_fault_tolerance_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
    )
    add_test(NAME unit_utils_fault_tolerance_${TEST_NAME}
             COMMAND unit_utils_fault_tolerance_${TEST_NAME})
endmacro()

# Add all fault tolerance tests
add_fault_tolerance_test(test_async_checkpoint)
set_tests_properties(unit_utils_fault_tolerance_test_async_checkpoint PROPERTIES TIMEOUT 90)
add_fault_tolerance_test(test_checkpoint)
add_fault_tolerance_test(test_checkpoint_pool)
add_fault_tolerance_test(test_diagnostics)
add_fault_tolerance_test(test_fast_recovery)
add_fault_tolerance_test(test_fault_event_bus)
add_fault_tolerance_test(test_health_monitor)
set_tests_properties(unit_utils_fault_tolerance_test_health_monitor PROPERTIES TIMEOUT 60)
add_fault_tolerance_test(test_lockfree_metrics)
add_fault_tolerance_test(test_metrics_aggregator)
add_fault_tolerance_test(test_recovery)
add_fault_tolerance_test(test_recovery_cache)
add_fault_tolerance_test(test_recovery_pool)
add_fault_tolerance_test(test_state_machine)

# Enhanced Fault Tolerance Tests (Phase FT-2: Immune System Integration)
add_fault_tolerance_test(test_distributed_fault_tolerance)
add_fault_tolerance_test(test_hierarchical_recovery)
add_fault_tolerance_test(test_recovery_evolution)
add_fault_tolerance_test(test_byzantine_fault_tolerance)
add_fault_tolerance_test(test_graceful_degradation)
add_fault_tolerance_test(test_recovery_observability)
add_fault_tolerance_test(test_chaos_engineering)
add_fault_tolerance_test(test_predictive_analysis)

# Health Agent USE Function Tests (Phase 1)
add_fault_tolerance_test(test_health_agent_use_functions)
set_tests_properties(unit_utils_fault_tolerance_test_health_agent_use_functions PROPERTIES TIMEOUT 120)

# Health Agent Connection Function Tests (Phase 2)
add_fault_tolerance_test(test_health_agent_connection_functions)
set_tests_properties(unit_utils_fault_tolerance_test_health_agent_connection_functions PROPERTIES TIMEOUT 120)

# Health Agent Consistency Function Tests (Phase 3)
add_fault_tolerance_test(test_health_agent_consistency_functions)
set_tests_properties(unit_utils_fault_tolerance_test_health_agent_consistency_functions PROPERTIES TIMEOUT 120)

# Cognitive Recovery Coordinator Tests (Brain-Driven Recovery Orchestration)
add_fault_tolerance_test(test_cognitive_recovery)
set_tests_properties(unit_utils_fault_tolerance_test_cognitive_recovery PROPERTIES TIMEOUT 120)

# Health Agent Neural Integration Tests (Phase 5.5: SNN/LNN Health Monitoring)
add_fault_tolerance_test(test_health_agent_neural_functions)
set_tests_properties(unit_utils_fault_tolerance_test_health_agent_neural_functions PROPERTIES TIMEOUT 120)

# Health Agent Behavioral Integration Tests (Phase 5.6: Dragonfly/Portia Health Monitoring)
add_fault_tolerance_test(test_health_agent_behavioral_functions)
set_tests_properties(unit_utils_fault_tolerance_test_health_agent_behavioral_functions PROPERTIES TIMEOUT 120)
