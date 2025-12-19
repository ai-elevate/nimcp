# Integration tests for glial substrate bridge
cmake_minimum_required(VERSION 3.14)

# Test executable
add_executable(integration_glial_substrate
    test_glial_substrate_integration.cpp
)

# Link libraries
target_link_libraries(integration_glial_substrate
    PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
)

# Include directories
target_include_directories(integration_glial_substrate
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Add test
add_test(
    NAME integration_glial_substrate
    COMMAND integration_glial_substrate --gtest_brief=1
)
