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
