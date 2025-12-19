# ============================================================================
# Cognitive Integration Tests
# ============================================================================

# Emotion-Cognition Integration Test
add_executable(integration_cognitive_emotion_integration
    test_emotion_integration.cpp
)

target_link_libraries(integration_cognitive_emotion_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_emotion_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_cognitive_emotion_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_emotion_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;emotion"
        TIMEOUT 120
)

# Working Memory + Positional Encoding Integration Test
add_executable(integration_cognitive_working_memory_pe
    test_working_memory_pe_integration.cpp
)

target_link_libraries(integration_cognitive_working_memory_pe
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_working_memory_pe
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_cognitive_working_memory_pe PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_working_memory_pe
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;working_memory;positional_encoding"
        TIMEOUT 120
)

# Note: Subdirectories (logic, introspection, fault_tolerance, reasoning, bio_async, mirror_neurons)
# are added directly from the main test/CMakeLists.txt to avoid duplicate target issues

# Free Energy Principle Enhancement integration tests
add_subdirectory(free_energy)
add_subdirectory(autobiographical_memory)

# Cognitive Substrate Bridges Integration
add_subdirectory(substrate)

# Curiosity - FEP Bridge Integration Test
add_executable(integration_cognitive_curiosity_fep_bridge
    curiosity/test_curiosity_fep_bridge_integration.cpp
)

target_link_libraries(integration_cognitive_curiosity_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_curiosity_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_cognitive_curiosity_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_curiosity_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;curiosity;fep"
        TIMEOUT 120
)
