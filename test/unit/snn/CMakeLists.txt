# ============================================================================
# SNN Unit Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for SNN unit tests
# WHY:  Verify correctness of spiking neural network components
# HOW:  Build GoogleTest executables for each SNN module
# ============================================================================

# Guard: Check GTest availability
if(NOT GTest_FOUND)
    message(STATUS "GTest not found - skipping SNN unit tests")
    return()
endif()

# Common link libraries for all SNN tests
set(SNN_TEST_LIBS
    nimcp_snn
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)

# Common include directories
set(SNN_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/snn
)

# ============================================================================
# Test 1: SNN Configuration
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_config.cpp)
    add_executable(unit_snn_test_snn_config
        test_snn_config.cpp
    )

    target_link_libraries(unit_snn_test_snn_config
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_config
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_config PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_config
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;config"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_config")
endif()

# ============================================================================
# Test 2: SNN Network
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_network.cpp)
    add_executable(unit_snn_test_snn_network
        test_snn_network.cpp
    )

    target_link_libraries(unit_snn_test_snn_network
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_network
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_network PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_network
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;network"
            TIMEOUT 120
    )

    message(STATUS "  Added SNN unit test: test_snn_network")
endif()

# ============================================================================
# Test 3: SNN Encoding/Decoding
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_encoding.cpp)
    add_executable(unit_snn_test_snn_encoding
        test_snn_encoding.cpp
    )

    target_link_libraries(unit_snn_test_snn_encoding
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_encoding
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_encoding PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_encoding
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;encoding"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_encoding")
endif()

# ============================================================================
# Test 4: SNN Training
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_training.cpp)
    add_executable(unit_snn_test_snn_training
        test_snn_training.cpp
    )

    target_link_libraries(unit_snn_test_snn_training
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_training
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_training PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_training
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;training"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_training")
endif()

# ============================================================================
# Test 5: SNN Bio-Async Integration
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_bio_async.cpp)
    add_executable(unit_snn_test_snn_bio_async
        test_snn_bio_async.cpp
    )

    target_link_libraries(unit_snn_test_snn_bio_async
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_bio_async
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_bio_async PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_bio_async
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;bio_async"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_bio_async")
endif()

# ============================================================================
# Test 6: SNN Immune Integration
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_immune.cpp)
    add_executable(unit_snn_test_snn_immune
        test_snn_immune.cpp
    )

    target_link_libraries(unit_snn_test_snn_immune
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_immune
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_immune PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_immune
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;immune"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_immune")
endif()

# ============================================================================
# Include bridges subdirectory
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bridges/CMakeLists.txt)
    add_subdirectory(bridges)
endif()

# ============================================================================
# Test 7: SNN Numerical Stability
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_numerical_stability.cpp)
    add_executable(unit_snn_test_snn_numerical_stability
        test_snn_numerical_stability.cpp
    )

    target_link_libraries(unit_snn_test_snn_numerical_stability
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_numerical_stability
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_numerical_stability PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_numerical_stability
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;numerical_stability"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_numerical_stability")
endif()

# ============================================================================
# Test 8: SNN Adaptation (shared AHP / Na-K pump state)
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_snn_adaptation.cpp)
    add_executable(unit_snn_test_snn_adaptation
        test_snn_adaptation.cpp
    )

    target_link_libraries(unit_snn_test_snn_adaptation
        PRIVATE ${SNN_TEST_LIBS}
    )

    target_include_directories(unit_snn_test_snn_adaptation
        PRIVATE ${SNN_TEST_INCLUDES}
    )

    target_compile_features(unit_snn_test_snn_adaptation PRIVATE cxx_std_17)

    gtest_discover_tests(unit_snn_test_snn_adaptation
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES
            LABELS "unit;snn;adaptation"
            TIMEOUT 60
    )

    message(STATUS "  Added SNN unit test: test_snn_adaptation")
endif()

# ============================================================================
# Future tests (uncomment as source files are added):
# ============================================================================
# - test_snn_population.cpp
# - test_snn_simulation.cpp

# Exception handling tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/exception/CMakeLists.txt")
    add_subdirectory(exception)
endif()
