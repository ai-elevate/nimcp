# ============================================================================
# SNN Regression Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for SNN regression tests
# WHY:  Verify SNN stability and prevent regressions
# HOW:  Build GoogleTest executable with deterministic tests
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping SNN regression tests")
    return()
endif()

# Common link libraries for SNN regression tests
set(SNN_REGRESSION_TEST_LIBS
    nimcp_snn
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)

# Common include directories
set(SNN_REGRESSION_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/snn
)

# ============================================================================
# SNN Regression Test
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_regression.cpp)
    add_executable(regression_snn_test_regression
        test_snn_regression.cpp
    )

    target_link_libraries(regression_snn_test_regression
        PRIVATE ${SNN_REGRESSION_TEST_LIBS}
    )

    target_include_directories(regression_snn_test_regression
        PRIVATE ${SNN_REGRESSION_TEST_INCLUDES}
    )

    target_compile_features(regression_snn_test_regression PRIVATE cxx_std_17)

    gtest_discover_tests(regression_snn_test_regression
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "regression;snn"
            TIMEOUT 120
    )

    message(STATUS "  Added SNN regression test: test_snn_regression")
endif()

# ============================================================================
# SNN Bridge Regression Test
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_bridge_regression.cpp)
    add_executable(regression_snn_test_bridge_regression
        test_snn_bridge_regression.cpp
    )

    target_link_libraries(regression_snn_test_bridge_regression
        PRIVATE ${SNN_REGRESSION_TEST_LIBS}
    )

    target_include_directories(regression_snn_test_bridge_regression
        PRIVATE ${SNN_REGRESSION_TEST_INCLUDES}
    )

    target_compile_features(regression_snn_test_bridge_regression PRIVATE cxx_std_17)

    gtest_discover_tests(regression_snn_test_bridge_regression
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "regression;snn;bridges"
            TIMEOUT 120
    )

    message(STATUS "  Added SNN bridge regression test: test_snn_bridge_regression")
endif()

# ============================================================================
# SNN Numerical Stability Regression Test
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_numerical_regression.cpp)
    add_executable(regression_snn_test_numerical_regression
        test_snn_numerical_regression.cpp
    )

    target_link_libraries(regression_snn_test_numerical_regression
        PRIVATE ${SNN_REGRESSION_TEST_LIBS}
    )

    target_include_directories(regression_snn_test_numerical_regression
        PRIVATE ${SNN_REGRESSION_TEST_INCLUDES}
    )

    target_compile_features(regression_snn_test_numerical_regression PRIVATE cxx_std_17)

    gtest_discover_tests(regression_snn_test_numerical_regression
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "regression;snn;numerical"
            TIMEOUT 120
    )

    message(STATUS "  Added SNN numerical regression test: test_snn_numerical_regression")
endif()
