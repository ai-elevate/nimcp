#=============================================================================
# Middleware Integration Unit Tests CMakeLists
#=============================================================================

# Shannon Monitor Unit Tests
add_executable(unit_middleware_integration_test_shannon_monitor
    test_shannon_monitor.cpp
)

target_link_libraries(unit_middleware_integration_test_shannon_monitor
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME ShannonMonitorUnit
    COMMAND unit_middleware_integration_test_shannon_monitor
)

# Flow Tracker Unit Tests
add_executable(unit_middleware_integration_test_flow_tracker
    test_flow_tracker.cpp
)

target_link_libraries(unit_middleware_integration_test_flow_tracker
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME FlowTrackerUnit
    COMMAND unit_middleware_integration_test_flow_tracker
)

# Quantum Command Propagator Unit Tests (Phase 1.5.2)
add_executable(unit_middleware_integration_test_quantum_propagator
    test_quantum_command_propagator.cpp
)

target_link_libraries(unit_middleware_integration_test_quantum_propagator
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        nimcp_middleware
        pthread
        m
)

add_test(NAME QuantumPropagatorUnit
    COMMAND unit_middleware_integration_test_quantum_propagator
)

# Executive Middleware Adapter Unit Tests (Phase 1.5.2)
add_executable(unit_middleware_integration_test_executive_adapter
    test_executive_middleware_adapter.cpp
)

target_link_libraries(unit_middleware_integration_test_executive_adapter
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        nimcp_middleware
        pthread
        m
)

add_test(NAME ExecutiveAdapterUnit
    COMMAND unit_middleware_integration_test_executive_adapter
)
