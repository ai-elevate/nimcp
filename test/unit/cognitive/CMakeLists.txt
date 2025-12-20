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

# Cognitive Meta-Controller Tests
add_executable(unit_cognitive_test_meta_controller
    test_cognitive_meta_controller.cpp
)

target_link_libraries(unit_cognitive_test_meta_controller
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_meta_controller
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_meta_controller PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_meta_controller
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;meta_controller"
        TIMEOUT 120
)

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

# Free Energy Principle Enhancement tests
add_subdirectory(free_energy)

# Autobiographical Memory tests
add_subdirectory(autobiographical_memory)

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

# Curiosity - FEP Bridge Tests
add_executable(unit_cognitive_test_curiosity_fep_bridge
    curiosity/test_curiosity_fep_bridge.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_curiosity_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;fep"
        TIMEOUT 120
)

# ============================================================================
# Enhanced Curiosity Tests (10 Enhancements)
# ============================================================================
# WHAT: Unit tests for Enhanced Curiosity System
# WHY:  Validate 10 curiosity enhancements (boredom, interest decay, types,
#       anxiety balance, social, meta, contagion, surprise, fatigue, counterfactual)
# HOW:  Test each enhancement plus integration, thread safety, edge cases
# ============================================================================

add_executable(unit_cognitive_test_curiosity_enhanced
    curiosity/test_curiosity_enhanced.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_enhanced
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_enhanced
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_curiosity_enhanced PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_enhanced
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;enhanced"
        TIMEOUT 180
)

# ============================================================================
# Cognitive FEP Bridge Tests (Comprehensive)
# ============================================================================
# WHAT: Unit tests for Cognitive-FEP bidirectional integration bridges
# WHY:  Ensure cognitive systems integrate with FEP predictions and precision
# HOW:  Test lifecycle, effects, and bio-async for consolidation, hierarchical,
#       meta-learning, personality, wellbeing, sleep-wake, and brain immune
# ============================================================================

add_executable(unit_cognitive_test_fep_bridges
    test_cognitive_fep_bridges.cpp
)

target_link_libraries(unit_cognitive_test_fep_bridges
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_fep_bridges
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_fep_bridges PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_fep_bridges
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;fep;bridge"
        TIMEOUT 120
)

# ============================================================================
# Sleep Bridge Integration Tests
# ============================================================================
# WHAT: Unit tests for Sleep-Cognitive integration bridges
# WHY:  Ensure cognitive functions correctly modulate based on sleep state
# HOW:  Test helper functions, default config, biological validity
# ============================================================================

# Attention Sleep Bridge Tests
add_executable(unit_cognitive_attention_sleep_bridge
    attention/sleep/test_attention_sleep_bridge.cpp
)
target_link_libraries(unit_cognitive_attention_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_attention_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_attention_sleep_bridge
    PROPERTIES LABELS "unit;cognitive;attention;sleep")

# Working Memory Sleep Bridge Tests
add_executable(unit_cognitive_working_memory_sleep_bridge
    working_memory/sleep/test_working_memory_sleep_bridge.cpp
)
target_link_libraries(unit_cognitive_working_memory_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_working_memory_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_working_memory_sleep_bridge
    PROPERTIES LABELS "unit;cognitive;working_memory;sleep")

# Executive Sleep Bridge Tests
add_executable(unit_cognitive_executive_sleep_bridge
    executive/sleep/test_executive_sleep_bridge.cpp
)
target_link_libraries(unit_cognitive_executive_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_executive_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_executive_sleep_bridge
    PROPERTIES LABELS "unit;cognitive;executive;sleep")

# ============================================================================
# Substrate Bridge Tests (Phase 6)
# ============================================================================
# WHAT: Unit tests for Neural Substrate integration bridges
# WHY:  Ensure cognitive systems correctly modulate based on metabolic state
# HOW:  Test ATP, temperature, fatigue effects on cognitive functions
# ============================================================================

# Attention tests (includes substrate bridge)
add_subdirectory(attention)

# Emotion tests (includes substrate bridge)
add_subdirectory(emotion)

# Executive tests (includes substrate bridge)
add_subdirectory(executive)

# Note: introspection and reasoning directories are added from test/CMakeLists.txt
# Their substrate subdirectories are included via those parent CMakeLists

# Memory tests (includes substrate bridge)
add_subdirectory(memory)

# Theory of Mind tests (includes substrate bridge)
add_subdirectory(tom)
