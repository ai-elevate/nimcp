# Unit tests for cognitive module bio-async integration

# Cognitive bio-async integration tests
add_executable(unit_cognitive_test_bio_async
    test_cognitive_bio_async.cpp
)

target_link_libraries(unit_cognitive_test_bio_async
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_bio_async
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_bio_async PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_bio_async
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;bio-async"
        TIMEOUT 90
)

# Cognitive bio-async registration tests
add_executable(unit_cognitive_test_bio_async_registration
    test_cognitive_bio_async_registration.cpp
)
target_link_libraries(unit_cognitive_test_bio_async_registration
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_cognitive_test_bio_async_registration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)
gtest_discover_tests(unit_cognitive_test_bio_async_registration)
