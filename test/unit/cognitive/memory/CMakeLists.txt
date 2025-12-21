# Memory Unit Tests

# Substrate bridge tests
add_subdirectory(substrate)

# Quantum Semantic Memory Tests
add_executable(unit_cognitive_memory_test_quantum_semantic
    test_quantum_semantic.cpp
)
target_link_libraries(unit_cognitive_memory_test_quantum_semantic
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_memory_test_quantum_semantic PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_memory_test_quantum_semantic
    PROPERTIES LABELS "unit;cognitive;memory;quantum")
