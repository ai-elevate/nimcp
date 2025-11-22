# Complex Math Unit Tests
# Phase 0.5: Complex Number Support for Neural Phase Coding

# Helper macro to add complex math test
macro(add_complex_math_test TEST_NAME)
    add_executable(unit_utils_math_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_utils_math_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m  # Math library for complex.h
    )
    add_test(NAME unit_utils_math_${TEST_NAME}
             COMMAND unit_utils_math_${TEST_NAME})
endmacro()

# Add all complex math tests
add_complex_math_test(test_complex_math)
