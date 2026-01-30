# ============================================================================
# Financial GPU Integration Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for financial GPU integration tests
# WHY:  Enable testing of financial GPU pipelines and workflows
# HOW:  Link against Check framework and nimcp library
# ============================================================================

# Guard: Check if Check framework is available
find_package(Check QUIET)
if(NOT Check_FOUND)
    message(STATUS "Check framework not found - skipping financial GPU integration tests")
    return()
endif()

# ============================================================================
# Test: Financial GPU Integration
# ============================================================================
add_executable(integration_financial_gpu
    test_financial_gpu_integration.c
)

target_link_libraries(integration_financial_gpu
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(integration_financial_gpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME integration_financial_gpu COMMAND integration_financial_gpu)
set_tests_properties(integration_financial_gpu PROPERTIES
    TIMEOUT 300
    LABELS "integration;gpu;financial"
)

# ============================================================================
# Test: Financial CPU/GPU Equivalence (if exists)
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_financial_cpu_gpu_equivalence.c)
    add_executable(integration_financial_cpu_gpu_equiv
        test_financial_cpu_gpu_equivalence.c
    )

    target_link_libraries(integration_financial_cpu_gpu_equiv
        PRIVATE
            nimcp
            Check::check
            m
            pthread
    )

    target_include_directories(integration_financial_cpu_gpu_equiv
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
    )

    add_test(NAME integration_financial_cpu_gpu_equiv COMMAND integration_financial_cpu_gpu_equiv)
    set_tests_properties(integration_financial_cpu_gpu_equiv PROPERTIES
        TIMEOUT 180
        LABELS "integration;gpu;financial;equivalence"
    )
endif()
