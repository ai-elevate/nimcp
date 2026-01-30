# ============================================================================
# Fuzzy GPU Integration Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for fuzzy GPU integration tests
# WHY:  Enable testing of fuzzy GPU inference and training pipelines
# HOW:  Link against Check framework and nimcp library
# ============================================================================

# Guard: Check if Check framework is available
find_package(Check QUIET)
if(NOT Check_FOUND)
    message(STATUS "Check framework not found - skipping fuzzy GPU integration tests")
    return()
endif()

# ============================================================================
# Test: Fuzzy GPU Inference Integration
# ============================================================================
add_executable(integration_fuzzy_gpu_inference
    test_fuzzy_gpu_inference.c
)

target_link_libraries(integration_fuzzy_gpu_inference
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(integration_fuzzy_gpu_inference
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME integration_fuzzy_gpu_inference COMMAND integration_fuzzy_gpu_inference)
set_tests_properties(integration_fuzzy_gpu_inference PROPERTIES
    TIMEOUT 180
    LABELS "integration;gpu;fuzzy"
)

# ============================================================================
# Test: Fuzzy GPU ANFIS Training Integration
# ============================================================================
add_executable(integration_fuzzy_gpu_anfis
    test_fuzzy_gpu_anfis.c
)

target_link_libraries(integration_fuzzy_gpu_anfis
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(integration_fuzzy_gpu_anfis
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME integration_fuzzy_gpu_anfis COMMAND integration_fuzzy_gpu_anfis)
set_tests_properties(integration_fuzzy_gpu_anfis PROPERTIES
    TIMEOUT 300
    LABELS "integration;gpu;fuzzy;training"
)

# ============================================================================
# Test: Fuzzy GPU Relation Composition (if exists)
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_fuzzy_gpu_relation.c)
    add_executable(integration_fuzzy_gpu_relation
        test_fuzzy_gpu_relation.c
    )

    target_link_libraries(integration_fuzzy_gpu_relation
        PRIVATE
            nimcp
            Check::check
            m
            pthread
    )

    target_include_directories(integration_fuzzy_gpu_relation
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
    )

    add_test(NAME integration_fuzzy_gpu_relation COMMAND integration_fuzzy_gpu_relation)
    set_tests_properties(integration_fuzzy_gpu_relation PROPERTIES
        TIMEOUT 120
        LABELS "integration;gpu;fuzzy"
    )
endif()
