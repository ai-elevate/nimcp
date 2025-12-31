# GPU Unit Tests

macro(add_gpu_test TEST_NAME)
    add_executable(unit_gpu_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_gpu_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
            pthread
    )
    add_test(NAME unit_gpu_${TEST_NAME}
             COMMAND unit_gpu_${TEST_NAME})
endmacro()

# GPU detection and execution tests
add_gpu_test(test_simd_detect)
add_gpu_test(test_network_detect)
add_gpu_test(test_execution_mode)
add_gpu_test(test_gpu_neuron)
add_gpu_test(test_spike_event)

# Include immune subdirectory tests
add_subdirectory(immune)
