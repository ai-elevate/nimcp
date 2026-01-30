# ============================================================================
# Financial GPU Integration Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for financial GPU integration tests
# WHY:  Enable testing of financial GPU pipelines and workflows
# HOW:  Link against GTest framework and nimcp library
# ============================================================================

# Find GTest
find_package(GTest QUIET)

if(GTest_FOUND AND CUDAToolkit_FOUND)
    message(STATUS "Adding financial GPU integration tests (GTest)...")

    # ========================================================================
    # Test: Financial GPU Integration
    # ========================================================================
    add_executable(integration_gpu_financial_test_integration
        test_financial_gpu_integration.cpp
    )

    target_link_libraries(integration_gpu_financial_test_integration
        PRIVATE
            nimcp
            GTest::gtest
            GTest::gtest_main
            ${CUDA_LIBRARIES}
            m
            pthread
    )

    target_include_directories(integration_gpu_financial_test_integration
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CUDA_INCLUDE_DIRS}
    )

    add_test(NAME integration_gpu_financial COMMAND integration_gpu_financial_test_integration)
    set_tests_properties(integration_gpu_financial PROPERTIES
        TIMEOUT 300
        LABELS "integration;gpu;financial"
    )

else()
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found - skipping financial GPU integration tests")
    endif()
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "CUDA toolkit not found - skipping financial GPU integration tests")
    endif()
endif()

# ============================================================================
# Legacy Check-based tests (if available)
# ============================================================================
find_package(Check QUIET)
if(Check_FOUND)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_financial_gpu_integration.c)
        add_executable(integration_financial_gpu_check
            test_financial_gpu_integration.c
        )
        target_link_libraries(integration_financial_gpu_check
            PRIVATE nimcp Check::check m pthread
        )
        target_include_directories(integration_financial_gpu_check
            PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
        )
        add_test(NAME integration_financial_gpu_check COMMAND integration_financial_gpu_check)
    endif()
endif()
