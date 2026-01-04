# Hypothalamus Region Unit Tests
# Includes homeostatic regulation, circadian rhythms, stress response, autonomic control

# Helper macro to add Hypothalamus test
macro(add_hypothalamus_test TEST_NAME)
    add_executable(unit_core_brain_regions_hypothalamus_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_core_brain_regions_hypothalamus_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m  # Math library
            pthread
    )
    add_test(NAME unit_core_brain_regions_hypothalamus_${TEST_NAME}
             COMMAND unit_core_brain_regions_hypothalamus_${TEST_NAME})
endmacro()

# Core Hypothalamus adapter tests (Phase H1)
add_hypothalamus_test(test_nimcp_hypothalamus_adapter)

# Alignment hardening tests (Phase 19)
add_hypothalamus_test(test_nimcp_hypothalamus_alignment)
