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

# CUDA Kernel tests - only built when CUDA toolkit is available
# These tests require actual CUDA kernel implementations
if(CUDAToolkit_FOUND)
    message(STATUS "  Building CUDA kernel unit tests (CUDA toolkit found)")

    # GPU quantum and inference kernel tests
    add_gpu_test(test_quantum_kernels)
    add_gpu_test(test_inference_kernels)

    # GPU tensor and training kernel tests
    add_gpu_test(test_tensor_kernels)
    add_gpu_test(test_training_kernels)

    # GPU SNN and LNN kernel tests
    add_gpu_test(test_snn_kernels)
    add_gpu_test(test_lnn_kernels)
    add_gpu_test(test_lnn_ode_kernels)

    # GPU Swarm Intelligence kernel tests
    add_gpu_test(test_swarm_kernels)

    # GPU Neural Substrate kernel tests (axon, dendrite, myelin, neuromod, glial, metabolic)
    add_gpu_test(test_substrate_gpu)
else()
    message(STATUS "  Skipping CUDA kernel tests (CUDA toolkit not found)")
endif()

# Include subdirectory tests
add_subdirectory(immune)

# Include cognitive subdirectory tests (Broca GPU, etc.)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cognitive/CMakeLists.txt)
    add_subdirectory(cognitive)
endif()
