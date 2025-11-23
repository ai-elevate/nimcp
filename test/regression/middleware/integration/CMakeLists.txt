# Regression tests for middleware controller (Phase 1.5.5)

add_executable(regression_middleware_controller
    test_middleware_controller_regression.cpp
)

target_link_libraries(regression_middleware_controller
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
    m
)

target_include_directories(regression_middleware_controller PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
)

add_test(NAME regression_middleware_controller
    COMMAND regression_middleware_controller)
