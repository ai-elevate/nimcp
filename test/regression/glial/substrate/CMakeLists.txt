# Regression tests for glial substrate bridge
cmake_minimum_required(VERSION 3.14)

# Test executable
add_executable(regression_glial_substrate
    test_glial_substrate_regression.cpp
)

# Link libraries
target_link_libraries(regression_glial_substrate
    PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
)

# Include directories
target_include_directories(regression_glial_substrate
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Add test
add_test(
    NAME regression_glial_substrate
    COMMAND regression_glial_substrate --gtest_brief=1
)
