# ============================================================================
# Quantum Bridge Regression Tests
# ============================================================================
# WHAT: Regression tests for quantum bridges stability and consistency
# WHY:  Ensure quantum bridge behavior doesn't regress over time
# HOW:  Test edge cases, boundary conditions, and behavioral consistency

add_executable(regression_quantum_bridges
    test_quantum_bridges_regression.cpp
)

target_include_directories(regression_quantum_bridges PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(regression_quantum_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        m
        pthread
)

target_compile_features(regression_quantum_bridges PRIVATE cxx_std_17)

add_test(NAME regression_quantum_bridges COMMAND regression_quantum_bridges)
set_tests_properties(regression_quantum_bridges PROPERTIES LABELS "regression;quantum")
