# ============================================================================
# Financial GPU Unit Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for financial GPU unit tests
# WHY:  Enable testing of financial GPU kernels
# HOW:  Link against Check framework and nimcp library
# ============================================================================

# Guard: Check if Check framework is available
find_package(Check QUIET)
if(NOT Check_FOUND)
    message(STATUS "Check framework not found - skipping financial GPU unit tests")
    return()
endif()

# ============================================================================
# Test: Financial GPU Monte Carlo
# ============================================================================
add_executable(test_financial_monte_carlo_gpu
    test_financial_monte_carlo_gpu.c
)

target_link_libraries(test_financial_monte_carlo_gpu
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(test_financial_monte_carlo_gpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_financial_monte_carlo_gpu COMMAND test_financial_monte_carlo_gpu)
set_tests_properties(unit_financial_monte_carlo_gpu PROPERTIES
    TIMEOUT 120
    LABELS "unit;gpu;financial"
)

# ============================================================================
# Test: Financial GPU Optimization
# ============================================================================
add_executable(test_financial_optimization_gpu
    test_financial_optimization_gpu.c
)

target_link_libraries(test_financial_optimization_gpu
    PRIVATE
        nimcp
        Check::check
        m
        pthread
)

target_include_directories(test_financial_optimization_gpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_financial_optimization_gpu COMMAND test_financial_optimization_gpu)
set_tests_properties(unit_financial_optimization_gpu PROPERTIES
    TIMEOUT 120
    LABELS "unit;gpu;financial"
)

# ============================================================================
# Test: Financial GPU Risk
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_financial_risk_gpu.c)
    add_executable(test_financial_risk_gpu
        test_financial_risk_gpu.c
    )

    target_link_libraries(test_financial_risk_gpu
        PRIVATE
            nimcp
            Check::check
            m
            pthread
    )

    target_include_directories(test_financial_risk_gpu
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
    )

    add_test(NAME unit_financial_risk_gpu COMMAND test_financial_risk_gpu)
    set_tests_properties(unit_financial_risk_gpu PROPERTIES
        TIMEOUT 60
        LABELS "unit;gpu;financial"
    )
endif()

# ============================================================================
# Test: Financial GPU Derivatives
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_financial_derivatives_gpu.c)
    add_executable(test_financial_derivatives_gpu
        test_financial_derivatives_gpu.c
    )

    target_link_libraries(test_financial_derivatives_gpu
        PRIVATE
            nimcp
            Check::check
            m
            pthread
    )

    target_include_directories(test_financial_derivatives_gpu
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
    )

    add_test(NAME unit_financial_derivatives_gpu COMMAND test_financial_derivatives_gpu)
    set_tests_properties(unit_financial_derivatives_gpu PROPERTIES
        TIMEOUT 60
        LABELS "unit;gpu;financial"
    )
endif()
