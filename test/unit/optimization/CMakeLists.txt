# ============================================================================
# 40-Watt Brain Optimization Unit Tests
# ============================================================================
# Uses shared brain fixtures (one brain per test suite, not per test)
# for efficient test execution

# Active Neuron Tracking (Phase 1)
add_executable(unit_active_neuron_tracking
    test_active_neuron_tracking.cpp
)
target_link_libraries(unit_active_neuron_tracking
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_active_neuron_tracking
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(unit_active_neuron_tracking PRIVATE cxx_std_17)
gtest_discover_tests(unit_active_neuron_tracking
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "unit;optimization;active_neurons" TIMEOUT 120
)

# Learning Workspace (Phase 2)
add_executable(unit_learning_workspace
    test_learning_workspace.cpp
)
target_link_libraries(unit_learning_workspace
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_learning_workspace
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(unit_learning_workspace PRIVATE cxx_std_17)
gtest_discover_tests(unit_learning_workspace
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "unit;optimization;workspace" TIMEOUT 120
)

# Fast Math (Phase 3)
add_executable(unit_fast_math
    test_fast_math.cpp
)
target_link_libraries(unit_fast_math
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_fast_math
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(unit_fast_math PRIVATE cxx_std_17)
gtest_discover_tests(unit_fast_math
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "unit;optimization;fast_math" TIMEOUT 120
)

# Lazy Init (Phase 4)
add_executable(unit_lazy_init
    test_lazy_init.cpp
)
target_link_libraries(unit_lazy_init
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_lazy_init
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(unit_lazy_init PRIVATE cxx_std_17)
gtest_discover_tests(unit_lazy_init
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "unit;optimization;lazy_init" TIMEOUT 120
)

# Synapse Fast Path (Phase 5)
add_executable(unit_synapse_fast_path
    test_synapse_fast_path.cpp
)
target_link_libraries(unit_synapse_fast_path
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_synapse_fast_path
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(unit_synapse_fast_path PRIVATE cxx_std_17)
gtest_discover_tests(unit_synapse_fast_path
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "unit;optimization;synapse_fast" TIMEOUT 120
)
