# ============================================================================
# Cognitive Regression Tests
# ============================================================================

# Emotion-Cognition Performance Regression Test
add_executable(regression_cognitive_emotion_performance
    test_emotion_performance.cpp
)

target_link_libraries(regression_cognitive_emotion_performance
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(regression_cognitive_emotion_performance
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(regression_cognitive_emotion_performance PRIVATE cxx_std_17)

gtest_discover_tests(regression_cognitive_emotion_performance
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "regression;cognitive;emotion;performance"
        TIMEOUT 180
)

# Positional encoding regression tests for working memory
add_executable(regression_cognitive_working_memory_pe_test
    test_working_memory_pe_regression.cpp
)
target_link_libraries(regression_cognitive_working_memory_pe_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(regression_cognitive_working_memory_pe_test PRIVATE cxx_std_17)
gtest_discover_tests(regression_cognitive_working_memory_pe_test
    PROPERTIES
        TIMEOUT 600
        LABELS "regression;cognitive;working_memory;positional_encoding"
)

# Add subdirectories for specialized regression tests
add_subdirectory(logic)
add_subdirectory(introspection)
add_subdirectory(fault_tolerance)
add_subdirectory(memory)
add_subdirectory(reasoning)
add_subdirectory(bio_async)
add_subdirectory(consolidation)
add_subdirectory(emotions)
add_subdirectory(emotion_tensor)
add_subdirectory(ethics)
add_subdirectory(executive)
add_subdirectory(global_workspace)
add_subdirectory(grief)
add_subdirectory(joy)
add_subdirectory(mirror_neurons)
# Brain Immune System tests - Added via test/CMakeLists.txt to avoid duplicate
# add_subdirectory(immune)
