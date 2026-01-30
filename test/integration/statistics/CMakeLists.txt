#=============================================================================
# CMakeLists.txt - Statistics Integration Tests
#=============================================================================

# Integration test macro
macro(add_statistics_integration_test TEST_NAME)
    add_executable(integration_statistics_${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(integration_statistics_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m
    )
    target_include_directories(integration_statistics_${TEST_NAME}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_features(integration_statistics_${TEST_NAME} PRIVATE cxx_std_17)
    add_test(
        NAME integration_statistics_${TEST_NAME}
        COMMAND integration_statistics_${TEST_NAME}
    )
endmacro()

# Integration tests
add_statistics_integration_test(test_statistics_shannon_integration)
