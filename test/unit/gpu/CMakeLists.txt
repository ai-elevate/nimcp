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

    # GPU Axon module tests
    add_gpu_test(test_axon_gpu)

    # GPU Dendrite module tests
    add_gpu_test(test_dendrite_gpu)

    # GPU Myelin module tests
    add_gpu_test(test_myelin_gpu)

    # GPU Brain Oscillation kernel tests (PLV, PLI, PAC, coherence, FFT)
    add_gpu_test(test_oscillations_kernels)

    # GPU Memory Consolidation kernel tests (hippocampal replay, systems consolidation, engrams)
    add_gpu_test(test_memory_consolidation_kernels)

    # GPU Knowledge Graph kernel tests (BFS/DFS traversal, similarity, hyperbolic space)
    add_gpu_test(test_knowledge_graph_kernels)

    # GPU Synaptic Transmission kernel tests (Tsodyks-Markram, receptor kinetics)
    add_gpu_test(test_synapse_kernels)

    # GPU Dragonfly Vision kernel tests (target tracking, optical flow, STMD)
    add_gpu_test(test_dragonfly_vision_kernels)

    # GPU Plasticity kernel tests (STDP, BCM, homeostatic, STP, calcium dynamics)
    add_gpu_test(test_plasticity_kernels)
else()
    message(STATUS "  Skipping CUDA kernel tests (CUDA toolkit not found)")
endif()

# Include subdirectory tests
add_subdirectory(immune)

# Include cognitive subdirectory tests (Broca GPU, etc.)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cognitive/CMakeLists.txt)
    add_subdirectory(cognitive)
endif()
