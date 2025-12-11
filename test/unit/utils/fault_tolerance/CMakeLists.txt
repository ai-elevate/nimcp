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
add_fault_tolerance_test(test_checkpoint)
add_fault_tolerance_test(test_checkpoint_pool)
add_fault_tolerance_test(test_diagnostics)
add_fault_tolerance_test(test_fast_recovery)
add_fault_tolerance_test(test_fault_event_bus)
add_fault_tolerance_test(test_health_monitor)
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
