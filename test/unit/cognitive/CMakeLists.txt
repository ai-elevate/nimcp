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

# Emotion-Attention Integration Tests
# DISABLED: Source file nimcp_emotion_attention.c not yet implemented
message(STATUS "  [SKIP] test_emotion_attention.cpp - source file not implemented")

# Emotion-Consolidation Integration Tests
# DISABLED: Source file nimcp_emotion_consolidation.c not yet implemented
message(STATUS "  [SKIP] test_emotion_consolidation.cpp - source file not implemented")

# Working Memory subdirectory (PE tests)
add_subdirectory(working_memory)

# Wellbeing tests
add_subdirectory(wellbeing)

# Mental Health tests
add_subdirectory(mental_health)

# Working Memory - Immune Integration Tests
add_executable(unit_cognitive_test_working_memory_immune
    test_working_memory_immune_integration.cpp
)

target_link_libraries(unit_cognitive_test_working_memory_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_working_memory_immune
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_working_memory_immune PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_working_memory_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;working_memory;immune"
        TIMEOUT 120
)

# Brain Immune System tests - Added via test/CMakeLists.txt to avoid duplicate
# add_subdirectory(immune)

# Predictive-Immune Integration tests
add_subdirectory(predictive_immune)

# Global Workspace - Immune Integration Tests
add_executable(unit_cognitive_test_global_workspace_immune
    global_workspace/test_global_workspace_immune_integration.cpp
)

target_link_libraries(unit_cognitive_test_global_workspace_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_global_workspace_immune
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_global_workspace_immune PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_global_workspace_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;global_workspace;immune"
        TIMEOUT 120
)

# Curiosity - Immune Integration Tests
add_executable(unit_cognitive_test_curiosity_immune
    curiosity/test_curiosity_immune_integration.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_immune
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_curiosity_immune PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;immune"
        TIMEOUT 120
)
