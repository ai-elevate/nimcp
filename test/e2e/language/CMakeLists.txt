# ============================================================================
# Grounded Language End-to-End Tests
# ============================================================================
add_executable(e2e_grounded_language
    test_grounded_language_e2e.cpp
)
target_link_libraries(e2e_grounded_language
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(e2e_grounded_language
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(e2e_grounded_language PRIVATE cxx_std_17)
gtest_discover_tests(e2e_grounded_language
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "e2e;language;grounded"
        TIMEOUT 300
)
