# ============================================================================
# Cognitive Integration Tests
# ============================================================================
# WHAT: Integration tests for Cognitive Integration bridges and hub
# WHY:  Validate cross-module integration flows and multi-bridge coordination
# HOW:  Test complete integration scenarios with multiple bridges working together
# ============================================================================

# Hub Module Integration Tests
add_executable(integration_cognitive_test_hub_module_integration
    test_hub_module_integration.cpp
)

target_link_libraries(integration_cognitive_test_hub_module_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_test_hub_module_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_cognitive_test_hub_module_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_test_hub_module_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;hub;module"
        TIMEOUT 180
)

# Emotion-Memory Integration Tests
add_executable(integration_cognitive_test_emotion_memory_integration
    test_emotion_memory_integration.cpp
)

target_link_libraries(integration_cognitive_test_emotion_memory_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_test_emotion_memory_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_cognitive_test_emotion_memory_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_test_emotion_memory_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;emotion;memory"
        TIMEOUT 180
)

# Attention-WM Integration Tests
add_executable(integration_cognitive_test_attention_wm_integration
    test_attention_wm_integration.cpp
)

target_link_libraries(integration_cognitive_test_attention_wm_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_test_attention_wm_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_cognitive_test_attention_wm_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_test_attention_wm_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;attention;wm"
        TIMEOUT 180
)

# Multi-Bridge Integration Tests
add_executable(integration_cognitive_test_multi_bridge_integration
    test_multi_bridge_integration.cpp
)

target_link_libraries(integration_cognitive_test_multi_bridge_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_test_multi_bridge_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_cognitive_test_multi_bridge_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_test_multi_bridge_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;multi;bridge"
        TIMEOUT 180
)

# Hub Orchestrator Integration Tests
add_executable(integration_cognitive_test_hub_orchestrator_integration
    test_hub_orchestrator_integration.cpp
)

target_link_libraries(integration_cognitive_test_hub_orchestrator_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_test_hub_orchestrator_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(integration_cognitive_test_hub_orchestrator_integration PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_test_hub_orchestrator_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;hub;orchestrator"
        TIMEOUT 180
)
