# Statistics Module Unit Tests
# Comprehensive statistical and probability methods testing

# Helper macro to add statistics test
macro(add_statistics_test TEST_NAME)
    add_executable(unit_utils_statistics_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_utils_statistics_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m  # Math library
    )
    target_compile_features(unit_utils_statistics_${TEST_NAME} PRIVATE cxx_std_17)
    add_test(NAME unit_utils_statistics_${TEST_NAME}
             COMMAND unit_utils_statistics_${TEST_NAME})
endmacro()

# Add all statistics tests
add_statistics_test(test_statistics)
add_statistics_test(test_quantum_statistics)
