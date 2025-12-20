# ============================================================================
# LNN Unit Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for LNN unit tests
# WHY:  Verify correctness of liquid neural network components
# HOW:  Build GoogleTest executables for each LNN module
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping LNN unit tests")
    return()
endif()

# Common link libraries for all LNN tests
set(LNN_TEST_LIBS
    nimcp_lnn
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)

# Common include directories
set(LNN_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/lnn
)

# ============================================================================
# Test 1: LNN Configuration
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_lnn_config.cpp)
    add_executable(unit_lnn_test_lnn_config
        test_lnn_config.cpp
    )

    target_link_libraries(unit_lnn_test_lnn_config
        PRIVATE ${LNN_TEST_LIBS}
    )

    target_include_directories(unit_lnn_test_lnn_config
        PRIVATE ${LNN_TEST_INCLUDES}
    )

    target_compile_features(unit_lnn_test_lnn_config PRIVATE cxx_std_17)

    gtest_discover_tests(unit_lnn_test_lnn_config
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;lnn;config"
            TIMEOUT 60
    )

    message(STATUS "  Added LNN unit test: test_lnn_config")
endif()

# ============================================================================
# Test 2: LNN Neuron
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_lnn_neuron.cpp)
    add_executable(unit_lnn_test_lnn_neuron
        test_lnn_neuron.cpp
    )

    target_link_libraries(unit_lnn_test_lnn_neuron
        PRIVATE ${LNN_TEST_LIBS}
    )

    target_include_directories(unit_lnn_test_lnn_neuron
        PRIVATE ${LNN_TEST_INCLUDES}
    )

    target_compile_features(unit_lnn_test_lnn_neuron PRIVATE cxx_std_17)

    gtest_discover_tests(unit_lnn_test_lnn_neuron
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;lnn;neuron"
            TIMEOUT 60
    )

    message(STATUS "  Added LNN unit test: test_lnn_neuron")
endif()

# ============================================================================
# Test 3: LNN Layer
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_lnn_layer.cpp)
    add_executable(unit_lnn_test_lnn_layer
        test_lnn_layer.cpp
    )

    target_link_libraries(unit_lnn_test_lnn_layer
        PRIVATE ${LNN_TEST_LIBS}
    )

    target_include_directories(unit_lnn_test_lnn_layer
        PRIVATE ${LNN_TEST_INCLUDES}
    )

    target_compile_features(unit_lnn_test_lnn_layer PRIVATE cxx_std_17)

    gtest_discover_tests(unit_lnn_test_lnn_layer
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;lnn;layer"
            TIMEOUT 120
    )

    message(STATUS "  Added LNN unit test: test_lnn_layer")
endif()

# ============================================================================
# Future tests (uncomment as source files are added):
# ============================================================================
# - test_lnn_network.cpp
# - test_lnn_ode.cpp
# - test_lnn_wiring.cpp
# - test_lnn_gradient.cpp
# - test_lnn_training.cpp
# - test_lnn_bio_async.cpp
# - test_lnn_immune.cpp
# - test_lnn_parallel.cpp
