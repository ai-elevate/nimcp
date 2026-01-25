# ============================================================================
# SNN Bridge E2E Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for SNN bridge end-to-end tests
# WHY:  Test full bridge lifecycle and multi-bridge workflows
# HOW:  Build GoogleTest executable linking all required modules
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping SNN bridge E2E tests")
    return()
endif()

# Common link libraries
set(SNN_E2E_LIBS
    nimcp_snn
    nimcp
    nimcp_middleware
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)

# Common include directories
set(SNN_E2E_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/snn
    ${CMAKE_SOURCE_DIR}/test
)

# ============================================================================
# SNN Bridge Exception E2E Test
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_bridge_exception_e2e.cpp)
    add_executable(e2e_snn_bridge_exception
        test_snn_bridge_exception_e2e.cpp
    )

    target_link_libraries(e2e_snn_bridge_exception
        PRIVATE ${SNN_E2E_LIBS}
    )

    target_include_directories(e2e_snn_bridge_exception
        PRIVATE ${SNN_E2E_INCLUDES}
    )

    target_compile_features(e2e_snn_bridge_exception PRIVATE cxx_std_17)

    target_compile_definitions(e2e_snn_bridge_exception PRIVATE
        NIMCP_TESTING
        NIMCP_E2E_TESTING
    )

    gtest_discover_tests(e2e_snn_bridge_exception
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "e2e;snn;bridges;exception"
            TIMEOUT 300
    )

    message(STATUS "  Added SNN bridge exception E2E test: test_snn_bridge_exception_e2e")
endif()
