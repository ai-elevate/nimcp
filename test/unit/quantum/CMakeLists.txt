# Quantum Bridge Integration Tests

add_executable(unit_quantum_bridges
    test_quantum_bridges.cpp
)

target_include_directories(unit_quantum_bridges PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(unit_quantum_bridges
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        m
        pthread
)

target_compile_features(unit_quantum_bridges PRIVATE cxx_std_17)

add_test(NAME unit_quantum_bridges COMMAND unit_quantum_bridges)
