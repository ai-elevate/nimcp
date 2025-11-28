# Integration tests for bio-async module

add_executable(integration_async_test_bio_async_integration
    test_bio_async_integration.cpp
)

target_link_libraries(integration_async_test_bio_async_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_async_test_bio_async_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_async_test_bio_async_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_async_test_bio_async_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;async"
        TIMEOUT 120
)
