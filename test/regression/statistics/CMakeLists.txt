#=============================================================================
# CMakeLists.txt - Statistics Regression Tests
#=============================================================================

# Regression test macro
macro(add_statistics_regression_test TEST_NAME)
    add_executable(regression_statistics_${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(regression_statistics_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
    )
    target_include_directories(regression_statistics_${TEST_NAME}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_features(regression_statistics_${TEST_NAME} PRIVATE cxx_std_17)
    add_test(
        NAME regression_statistics_${TEST_NAME}
        COMMAND regression_statistics_${TEST_NAME}
    )
endmacro()

# Regression tests
add_statistics_regression_test(test_statistics_regression)
