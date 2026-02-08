# ============================================================================
# GPU Regression Tests CMakeLists.txt
# ============================================================================
# WHAT: Regression tests for GPU operations
# WHY:  Prevent reintroduction of fixed GPU-related bugs
# HOW:  Test specific bug scenarios and backward compatibility
# ============================================================================

# Guard: CUDA toolkit required for GPU regression tests
if(NOT CUDAToolkit_FOUND)
    message(STATUS "  Skipping GPU regression tests (CUDA toolkit not found)")
    return()
endif()

message(STATUS "  Building GPU regression tests (CUDA toolkit found)")

# Helper macro for adding GPU regression tests
macro(add_gpu_regression_test TEST_NAME)
    add_executable(regression_gpu_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(regression_gpu_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
            pthread
    )
    target_compile_features(regression_gpu_${TEST_NAME} PRIVATE cxx_std_20)
    target_compile_definitions(regression_gpu_${TEST_NAME} PRIVATE NIMCP_TESTING)
    add_test(NAME regression_gpu_${TEST_NAME}
             COMMAND regression_gpu_${TEST_NAME})
    set_tests_properties(regression_gpu_${TEST_NAME}
        PROPERTIES
            LABELS "regression;gpu"
            TIMEOUT 120
    )
endmacro()

# =============================================================================
# GPU Neuron Regression Tests
# =============================================================================
add_gpu_regression_test(test_gpu_neuron_backward_compat)
add_gpu_regression_test(test_gpu_neuron_regression)

# =============================================================================
# Multi-GPU Regression Tests
# =============================================================================
add_gpu_regression_test(test_multigpu_regression)

# P2-13: Device ID bounds overflow regression test
add_gpu_regression_test(test_gpu_bounds_regression)

# =============================================================================
# GPU Spike Event Regression Tests
# =============================================================================
add_gpu_regression_test(test_gpu_spike_event_backward_compat)

# =============================================================================
# GPU Execution Mode Regression Tests
# =============================================================================
add_gpu_regression_test(test_gpu_execution_mode_backward_compat)

# =============================================================================
# GPU Broca Regression Tests
# =============================================================================
# WHAT: Regression tests for GPU-accelerated Broca's region
# WHY:  Prevent reintroduction of GPU Broca bugs (alignment, overflow, etc.)
# HOW:  Test specific bug scenarios and verify fixes remain stable
add_gpu_regression_test(test_broca_gpu_regression)

# =============================================================================
# GPU Substrate Regression Tests
# =============================================================================
# WHAT: Regression tests for GPU-accelerated neural substrate
# WHY:  Prevent reintroduction of substrate GPU bugs (numerical stability, memory)
# HOW:  Test numerical stability, GPU/CPU equivalence, edge cases
add_gpu_regression_test(test_substrate_gpu_regression)

# =============================================================================
# GPU Visual Processing Regression Tests
# =============================================================================
# WHAT: Regression tests for GPU visual processing (V1-V5, saliency)
# WHY:  Prevent reintroduction of visual processing bugs (numerical, API)
# HOW:  Test struct layouts, function signatures, edge cases, memory safety
add_gpu_regression_test(test_visual_processing_regression)

# Include broca subdirectory if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/broca/CMakeLists.txt)
    add_subdirectory(broca)
endif()

# Include fuzzy subdirectory if it exists (controls auto-discovery)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fuzzy/CMakeLists.txt)
    add_subdirectory(fuzzy)
endif()

# =============================================================================
# GPU Fuzzy Regression Tests
# =============================================================================
# WHAT: Regression tests for GPU-accelerated fuzzy logic operations
# WHY:  Prevent reintroduction of fuzzy GPU bugs (numerical, overflow, etc.)
# HOW:  Test specific bug scenarios, numerical stability, edge cases
# DISABLED: Test includes missing header gpu/nimcp_gpu_context.h
# TODO: Fix header path to gpu/context/nimcp_gpu_context.h
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_fuzzy_gpu_regression.cpp)
#     add_gpu_regression_test(test_fuzzy_gpu_regression)
# endif()

# =============================================================================
# GPU Financial Regression Tests
# =============================================================================
# WHAT: Regression tests for GPU-accelerated financial operations
# WHY:  Prevent reintroduction of financial GPU bugs (Monte Carlo, options, etc.)
# HOW:  Test specific bug scenarios, numerical stability, edge cases
# DISABLED: Test has extern "C" linkage error - CUDA templates conflict with C linkage
# TODO: Fix extern "C" block in test_financial_gpu_regression.cpp line 17
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_financial_gpu_regression.cpp)
#     add_gpu_regression_test(test_financial_gpu_regression)
# endif()

# Include statistics GPU regression tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/statistics/CMakeLists.txt)
    add_subdirectory(statistics)
endif()

# Include recovery GPU regression tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/recovery/CMakeLists.txt)
    add_subdirectory(recovery)
endif()

# Include financial GPU regression tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/financial/CMakeLists.txt)
    add_subdirectory(financial)
endif()
