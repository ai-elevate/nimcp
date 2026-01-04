# Hypothalamus Integration Tests
# Tests alignment-drive integration, callback chains, thread safety

# Helper macro for hypothalamus integration tests
macro(add_hypothalamus_integration_test TEST_NAME)
    add_executable(integration_core_brain_regions_hypothalamus_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(integration_core_brain_regions_hypothalamus_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
            pthread
    )
    add_test(NAME integration_core_brain_regions_hypothalamus_${TEST_NAME}
             COMMAND integration_core_brain_regions_hypothalamus_${TEST_NAME})
endmacro()

# Alignment-Drive integration tests (Phase 19)
add_hypothalamus_integration_test(test_hypothalamus_alignment_integration)
