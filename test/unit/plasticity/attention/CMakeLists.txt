# Attention Positional Encoding Unit Tests
# Test coverage for attention PE integration

# Helper macro to add attention PE test
macro(add_attention_pe_test TEST_NAME)
    add_executable(unit_plasticity_attention_${TEST_NAME}
        ${TEST_NAME}.cpp
    )
    target_link_libraries(unit_plasticity_attention_${TEST_NAME}
        PRIVATE
            GTest::GTest
            GTest::Main
            nimcp
            m  # Math library
            pthread
    )
    add_test(NAME unit_plasticity_attention_${TEST_NAME}
             COMMAND unit_plasticity_attention_${TEST_NAME})
endmacro()

# Attention PE tests
add_attention_pe_test(test_attention_pe)

# Quantum Attention tests
add_attention_pe_test(test_quantum_attention)

# Exception Handling tests
add_attention_pe_test(test_attention_exception_handling)
