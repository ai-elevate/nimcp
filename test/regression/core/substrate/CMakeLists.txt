# CMakeLists.txt for Core Substrate Bridges Regression Tests

# Enable testing
enable_testing()

# Find GTest
find_package(GTest REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)

# Core substrate bridges regression test
add_executable(regression_core_substrate_bridges
    test_core_substrate_bridges_regression.cpp
)

target_link_libraries(regression_core_substrate_bridges
    nimcp
    GTest::GTest
    GTest::Main
    pthread
)

# Add test to CTest
add_test(
    NAME regression_core_substrate_bridges
    COMMAND regression_core_substrate_bridges --gtest_brief=1
)

# Set test properties
set_tests_properties(regression_core_substrate_bridges PROPERTIES
    TIMEOUT 300
    LABELS "regression;core;substrate;bridges"
)

# Optionally set working directory
set_tests_properties(regression_core_substrate_bridges PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Configured core substrate bridges regression tests")
