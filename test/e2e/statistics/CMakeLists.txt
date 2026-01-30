#=============================================================================
# CMakeLists.txt - Statistics End-to-End Tests
#=============================================================================

# E2E test macro
macro(add_statistics_e2e_test TEST_NAME)
    add_executable(e2e_statistics_${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(e2e_statistics_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
    )
    target_include_directories(e2e_statistics_${TEST_NAME}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_features(e2e_statistics_${TEST_NAME} PRIVATE cxx_std_17)
    add_test(
        NAME e2e_statistics_${TEST_NAME}
        COMMAND e2e_statistics_${TEST_NAME}
    )
endmacro()

# E2E tests
add_statistics_e2e_test(test_statistics_e2e)
