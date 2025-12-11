# Brain Immune System Unit Tests

# Brain Immune Unit Tests
add_executable(unit_cognitive_immune_brain_immune
    test_brain_immune.cpp
)
target_link_libraries(unit_cognitive_immune_brain_immune
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_brain_immune
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/security
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
)
add_test(NAME unit_cognitive_immune_brain_immune
         COMMAND unit_cognitive_immune_brain_immune)
set_tests_properties(unit_cognitive_immune_brain_immune PROPERTIES LABELS "unit")

# Perception Immune Integration Unit Tests
add_executable(unit_cognitive_immune_perception_immune
    test_perception_immune.cpp
)
target_link_libraries(unit_cognitive_immune_perception_immune
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_perception_immune
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/perception
)
add_test(NAME unit_cognitive_immune_perception_immune
         COMMAND unit_cognitive_immune_perception_immune)
set_tests_properties(unit_cognitive_immune_perception_immune PROPERTIES LABELS "unit")

# Immune-Plasticity Integration Unit Tests
add_executable(unit_cognitive_immune_plasticity_modulation
    unit_cognitive_immune_plasticity_modulation.cpp
)
target_link_libraries(unit_cognitive_immune_plasticity_modulation
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_plasticity_modulation
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/plasticity
)
add_test(NAME unit_cognitive_immune_plasticity_modulation
         COMMAND unit_cognitive_immune_plasticity_modulation)
set_tests_properties(unit_cognitive_immune_plasticity_modulation PROPERTIES LABELS "unit")

# Memory-Immune Integration Unit Tests
add_executable(unit_cognitive_immune_memory_integration
    unit_cognitive_immune_memory_integration.cpp
)
target_link_libraries(unit_cognitive_immune_memory_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_memory_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive
)
add_test(NAME unit_cognitive_immune_memory_integration
         COMMAND unit_cognitive_immune_memory_integration)
set_tests_properties(unit_cognitive_immune_memory_integration PROPERTIES LABELS "unit")

# Emotion-Immune Bridge Unit Tests
add_executable(unit_cognitive_immune_emotion_bridge
    unit_cognitive_immune_emotion_bridge.cpp
)
target_link_libraries(unit_cognitive_immune_emotion_bridge
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_emotion_bridge
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive
)
add_test(NAME unit_cognitive_immune_emotion_bridge
         COMMAND unit_cognitive_immune_emotion_bridge)
set_tests_properties(unit_cognitive_immune_emotion_bridge PROPERTIES LABELS "unit")

# Memory-Immune Subsystem Integration Unit Tests
add_executable(unit_cognitive_immune_memory_subsystem_integration
    test_memory_immune_subsystem_integration.cpp
)
target_link_libraries(unit_cognitive_immune_memory_subsystem_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_memory_subsystem_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive/memory
        ${CMAKE_SOURCE_DIR}/src/cognitive
)
add_test(NAME unit_cognitive_immune_memory_subsystem_integration
         COMMAND unit_cognitive_immune_memory_subsystem_integration)
set_tests_properties(unit_cognitive_immune_memory_subsystem_integration PROPERTIES LABELS "unit")

# Full Immune-Plasticity Integration Unit Tests (All Plasticity Modules)
add_executable(unit_cognitive_immune_plasticity_full
    test_immune_plasticity_full.cpp
)
target_link_libraries(unit_cognitive_immune_plasticity_full
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_plasticity_full
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/plasticity
        ${CMAKE_SOURCE_DIR}/src/plasticity/bcm
        ${CMAKE_SOURCE_DIR}/src/plasticity/stdp
        ${CMAKE_SOURCE_DIR}/src/plasticity/stp
        ${CMAKE_SOURCE_DIR}/src/plasticity/homeostatic
        ${CMAKE_SOURCE_DIR}/src/plasticity/dendritic
        ${CMAKE_SOURCE_DIR}/src/plasticity/adaptive
        ${CMAKE_SOURCE_DIR}/src/plasticity/eligibility
        ${CMAKE_SOURCE_DIR}/src/plasticity/predictive
)
add_test(NAME unit_cognitive_immune_plasticity_full
         COMMAND unit_cognitive_immune_plasticity_full)
set_tests_properties(unit_cognitive_immune_plasticity_full PROPERTIES LABELS "unit")

# Attention-Immune Bridge Unit Tests
add_executable(unit_cognitive_immune_attention_bridge
    test_attention_immune_integration.cpp
)
target_link_libraries(unit_cognitive_immune_attention_bridge
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_attention_bridge
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive/attention
        ${CMAKE_SOURCE_DIR}/src/plasticity/attention
)
add_test(NAME unit_cognitive_immune_attention_bridge
         COMMAND unit_cognitive_immune_attention_bridge)
set_tests_properties(unit_cognitive_immune_attention_bridge PROPERTIES LABELS "unit")

# Reasoning-Immune Bridge Unit Tests
add_executable(unit_cognitive_immune_reasoning_bridge
    test_reasoning_immune.cpp
)
target_link_libraries(unit_cognitive_immune_reasoning_bridge
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_immune_reasoning_bridge
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/src/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/cognitive/reasoning
)
add_test(NAME unit_cognitive_immune_reasoning_bridge
         COMMAND unit_cognitive_immune_reasoning_bridge)
set_tests_properties(unit_cognitive_immune_reasoning_bridge PROPERTIES LABELS "unit")
