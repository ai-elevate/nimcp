# Regression tests for async module (bio-async and traditional futures)

# =============================================================================
# Traditional Future Regression Tests
# =============================================================================
add_executable(regression_async_test_future_regression
    test_future_regression.cpp
)

target_link_libraries(regression_async_test_future_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_async_test_future_regression
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_async_test_future_regression PRIVATE cxx_std_17)

gtest_discover_tests(regression_async_test_future_regression
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;async"
        TIMEOUT 120
)

# =============================================================================
# Bio-Async Regression Tests
# =============================================================================
add_executable(regression_async_test_bio_async_regression
    test_bio_async_regression.cpp
)

target_link_libraries(regression_async_test_bio_async_regression
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_async_test_bio_async_regression
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_async_test_bio_async_regression PRIVATE cxx_std_17)

gtest_discover_tests(regression_async_test_bio_async_regression
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;async"
        TIMEOUT 120
)

# =============================================================================
# Wiring Diagram Regression Tests
# =============================================================================
add_executable(regression_async_test_wiring_diagram
    test_wiring_diagram_regression.cpp
)

target_link_libraries(regression_async_test_wiring_diagram
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_async_test_wiring_diagram
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_async_test_wiring_diagram PRIVATE cxx_std_17)

gtest_discover_tests(regression_async_test_wiring_diagram
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;async;wiring"
        TIMEOUT 120
)
