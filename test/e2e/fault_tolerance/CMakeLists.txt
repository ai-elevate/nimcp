# E2E Fault Tolerance Tests
# Health Agent Connection E2E Tests (Phase 2)

# Helper macro to add e2e test
macro(add_e2e_fault_tolerance_test TEST_NAME)
    add_executable(e2e_fault_tolerance_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(e2e_fault_tolerance_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            pthread
    )
    target_include_directories(e2e_fault_tolerance_${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME e2e_fault_tolerance_${TEST_NAME}
             COMMAND e2e_fault_tolerance_${TEST_NAME})
    set_tests_properties(e2e_fault_tolerance_${TEST_NAME} PROPERTIES LABELS "e2e")
endmacro()

# Health Agent Connection E2E Tests (Phase 2)
add_e2e_fault_tolerance_test(test_health_agent_connection_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_agent_connection_e2e PROPERTIES TIMEOUT 300)

# Health Agent Consistency E2E Tests (Phase 3)
add_e2e_fault_tolerance_test(test_health_agent_consistency_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_agent_consistency_e2e PROPERTIES TIMEOUT 300)

# Health Agent Active Cognitive Integration E2E Tests (Phase 5)
add_e2e_fault_tolerance_test(test_health_agent_active_cognitive_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_agent_active_cognitive_e2e PROPERTIES TIMEOUT 600)

# Health Agent Neural Integration E2E Tests (Phase 5.5: SNN/LNN)
add_e2e_fault_tolerance_test(test_health_agent_neural_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_agent_neural_e2e PROPERTIES TIMEOUT 300)

# Health Agent Behavioral Integration E2E Tests (Phase 5.6: Dragonfly/Portia)
add_e2e_fault_tolerance_test(test_health_agent_behavioral_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_agent_behavioral_e2e PROPERTIES TIMEOUT 300)

# Health Agent Memory System E2E Tests (Phase 5.7: Hippocampus/Mammillary)
add_e2e_fault_tolerance_test(test_health_agent_memory_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_agent_memory_e2e PROPERTIES TIMEOUT 300)

# Capacity Manager E2E Tests (Phase 5.8: Dynamic Capacity Management)
add_e2e_fault_tolerance_test(test_capacity_manager_e2e)
set_tests_properties(e2e_fault_tolerance_test_capacity_manager_e2e PROPERTIES TIMEOUT 300)

# Symbolic Logic Health E2E Tests (Phase 5.9: Symbolic Logic Health Integration)
add_e2e_fault_tolerance_test(test_symbolic_logic_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_symbolic_logic_health_e2e PROPERTIES TIMEOUT 300)

# Neural Substrate Health E2E Tests (Phase 5.10: Neural Substrate Health Integration)
add_e2e_fault_tolerance_test(test_substrate_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_substrate_health_e2e PROPERTIES TIMEOUT 300)

# Thalamic/Middleware Health E2E Tests (Phase 5.11: Thalamic/Middleware Health Integration)
add_e2e_fault_tolerance_test(test_thalamic_middleware_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_thalamic_middleware_health_e2e PROPERTIES TIMEOUT 300)

# Perception/Cortical Health E2E Tests (Phase 5.12: Perception/Cortical Health Integration)
add_e2e_fault_tolerance_test(test_perception_cortical_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_perception_cortical_health_e2e PROPERTIES TIMEOUT 300)

# Brain Probe Health E2E Tests (Phase 5.13: Brain Probes Enhancement)
add_e2e_fault_tolerance_test(test_brain_probe_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_brain_probe_health_e2e PROPERTIES TIMEOUT 300)

# World Model & Imagination Health E2E Tests (Phase 5.14: World Model & Imagination Health Integration)
add_e2e_fault_tolerance_test(test_world_imagination_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_world_imagination_health_e2e PROPERTIES TIMEOUT 300)

# Crash Recovery E2E Tests (Phase 6: End-to-End Testing)
add_e2e_fault_tolerance_test(test_crash_recovery)
set_tests_properties(e2e_fault_tolerance_test_crash_recovery PROPERTIES TIMEOUT 300)

# Resilience Stress E2E Tests (Phase 6: End-to-End Testing)
add_e2e_fault_tolerance_test(test_resilience_stress)
set_tests_properties(e2e_fault_tolerance_test_resilience_stress PROPERTIES TIMEOUT 600)

# Exception Resilience E2E Tests (Phase 6: Exception System Stress Testing)
add_e2e_fault_tolerance_test(test_exception_resilience_e2e)
set_tests_properties(e2e_fault_tolerance_test_exception_resilience_e2e PROPERTIES TIMEOUT 600)

# Exception Brain Integration E2E Tests (Phase 6: Exception-Brain Integration)
add_e2e_fault_tolerance_test(test_exception_brain_integration_e2e)
set_tests_properties(e2e_fault_tolerance_test_exception_brain_integration_e2e PROPERTIES TIMEOUT 300)

# Exception Flow E2E Tests (Complete Exception Handling Pipeline)
add_e2e_fault_tolerance_test(test_exception_flow_e2e)
set_tests_properties(e2e_fault_tolerance_test_exception_flow_e2e PROPERTIES TIMEOUT 300)

# Heartbeat Module E2E Tests (Phase 8: Heartbeat for Long Operations)
add_e2e_fault_tolerance_test(test_heartbeat_module_e2e)
set_tests_properties(e2e_fault_tolerance_test_heartbeat_module_e2e PROPERTIES TIMEOUT 300)

# State Checkpoint Recovery E2E Tests (Phase 8: System-Wide Health Integration)
add_e2e_fault_tolerance_test(test_state_checkpoint_recovery_e2e)
set_tests_properties(e2e_fault_tolerance_test_state_checkpoint_recovery_e2e PROPERTIES TIMEOUT 300)

# Self-Repair Bio-Async E2E Tests (Phase 8: Self-Repair Pipeline Bio-Async Integration)
add_e2e_fault_tolerance_test(test_self_repair_bio_async_e2e)
set_tests_properties(e2e_fault_tolerance_test_self_repair_bio_async_e2e PROPERTIES TIMEOUT 300)

# Health Self-Repair E2E Tests (Health Monitoring Self-Repair Integration)
add_e2e_fault_tolerance_test(test_health_self_repair_e2e)
set_tests_properties(e2e_fault_tolerance_test_health_self_repair_e2e PROPERTIES TIMEOUT 300)

# Plasticity Health E2E Tests (Plasticity Health Monitoring Pipeline)
add_e2e_fault_tolerance_test(test_plasticity_health_e2e)
set_tests_properties(e2e_fault_tolerance_test_plasticity_health_e2e PROPERTIES TIMEOUT 300)

# Exception Recovery E2E Tests (Complete Exception-to-Recovery Flow)
add_e2e_fault_tolerance_test(test_exception_recovery_e2e)
set_tests_properties(e2e_fault_tolerance_test_exception_recovery_e2e PROPERTIES TIMEOUT 600)

# Exception Full Pipeline E2E Tests - DISABLED (API issues)
# Tests moved to existing exception test suite

# Exception Immune Response E2E Tests - DISABLED (API issues)
# Tests moved to existing exception test suite
