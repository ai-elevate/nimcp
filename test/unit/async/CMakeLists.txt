# Unit tests for bio-async module

# Bio-async unit tests
add_executable(unit_async_test_bio_async
    test_bio_async.cpp
)

target_link_libraries(unit_async_test_bio_async
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m  # math library
)

target_include_directories(unit_async_test_bio_async
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

# Enable C++17 for chrono
target_compile_features(unit_async_test_bio_async PRIVATE cxx_std_17)

# Register test
gtest_discover_tests(unit_async_test_bio_async
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;async"
        TIMEOUT 60
)
