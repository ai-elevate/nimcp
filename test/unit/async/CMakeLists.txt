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

# Bio-router unit tests
add_executable(unit_async_test_bio_router
    test_bio_router.cpp
)

target_link_libraries(unit_async_test_bio_router
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_async_test_bio_router
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_async_test_bio_router PRIVATE cxx_std_17)

gtest_discover_tests(unit_async_test_bio_router
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;async;router"
        TIMEOUT 120
)

# Bio-async immune integration unit tests
add_executable(unit_async_immune_integration
    unit_async_immune_integration.cpp
)

target_link_libraries(unit_async_immune_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_async_immune_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_async_immune_integration PRIVATE cxx_std_17)

gtest_discover_tests(unit_async_immune_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;async;immune;integration"
        TIMEOUT 120
)
