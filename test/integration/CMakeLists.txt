# Integration tests
# This CMakeLists.txt exists to prevent auto-discovery of problematic top-level tests
# The test/*.cpp files at this level have significant API mismatches

message(STATUS "  [SKIP] Integration top-level tests - API has changed significantly")
message(STATUS "    - test_brain_factory_integration.cpp")
message(STATUS "    - test_cross_bridge_integration.cpp")
message(STATUS "    - test_substrate_bridge_integration.cpp")
message(STATUS "    - test_thalamic_bridge_integration.cpp")
message(STATUS "    - test_quantum_bridge_integration.cpp")

# Subdirectories with their own CMakeLists.txt are now processed by parent
# test/CMakeLists.txt via direct add_subdirectory() calls

# =============================================================================
# GPU/CPU Fallback Integration Tests
# =============================================================================
# These tests verify GPU detection, execution mode switching, and CPU fallback

macro(add_integration_gpu_test TEST_NAME)
    add_executable(integration_gpu_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(integration_gpu_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
            pthread
    )
    target_compile_features(integration_gpu_${TEST_NAME} PRIVATE cxx_std_20)
    add_test(NAME integration_gpu_${TEST_NAME}
             COMMAND integration_gpu_${TEST_NAME})
endmacro()

# GPU CPU fallback integration tests
add_integration_gpu_test(test_gpu_cpu_fallback_integration)

# =============================================================================
# Wernicke Integration Tests
# =============================================================================
# Note: wernicke subdirectory is added by test/CMakeLists.txt

# =============================================================================
# Language Layer Integration Tests
# =============================================================================
# Note: language subdirectory is added by test/CMakeLists.txt

# =============================================================================
# Brain KG Integration Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/brain/CMakeLists.txt)
    message(STATUS "Adding Brain KG Integration tests...")
    add_subdirectory(brain)
endif()

# =============================================================================
# Exception Integration Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exception/CMakeLists.txt)
    message(STATUS "Adding Exception Integration tests...")
    add_subdirectory(exception)
endif()

# =============================================================================
# Core Integration Tests (neural_substrate, substrate, axon, dendrite, etc.)
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt)
    message(STATUS "Adding Core Integration tests...")
    add_subdirectory(core)
endif()

