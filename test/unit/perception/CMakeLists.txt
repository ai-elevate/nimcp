# Speech Cortex Positional Encoding Unit Tests
# Test coverage for speech cortex PE integration

# Helper macro to add speech cortex PE test
macro(add_speech_cortex_pe_test TEST_NAME)
    add_executable(unit_perception_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_perception_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m  # Math library
            pthread
    )
    add_test(NAME unit_perception_${TEST_NAME}
             COMMAND unit_perception_${TEST_NAME})
endmacro()

# Speech Cortex PE tests
add_speech_cortex_pe_test(test_speech_cortex_pe)
