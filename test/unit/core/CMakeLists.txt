# Core Brain Regions Unit Tests

# Brain Regions subdirectory (PE tests)
add_subdirectory(brain_regions)

# Core Directives subdirectory (Action history, Asimov's Laws integration)
add_subdirectory(directives)

# Medulla Oblongata subdirectory (Arousal, Protection, Circadian)
add_subdirectory(medulla)

# Wernicke's Area subdirectory (Language comprehension)
# Note: wernicke subdirectory is added by test/CMakeLists.txt

# ============================================================================
# Sleep Bridge Integration Tests
# ============================================================================
# WHAT: Unit tests for Sleep-Core integration bridges
# WHY:  Ensure core brain systems correctly modulate based on sleep state
# HOW:  Test helper functions, default config, biological validity
# ============================================================================

# Oscillations Sleep Bridge Tests
add_executable(unit_core_oscillations_sleep_bridge
    brain_oscillations/sleep/test_oscillations_sleep_bridge.cpp
)
target_link_libraries(unit_core_oscillations_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_core_oscillations_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_core_oscillations_sleep_bridge
    PROPERTIES LABELS "unit;core;oscillations;sleep")

# ============================================================================
# Brain KG Message Index Tests (Phase 6: KG Query Optimization)
# ============================================================================
# WHAT: Unit tests for message-type index in brain knowledge graph
# WHY:  Verify efficient O(1) handler lookup by message type
# HOW:  Test CRUD operations on message-type index
# ============================================================================

add_executable(unit_core_brain_kg_message_index
    brain/test_brain_kg_message_index.cpp
)
target_link_libraries(unit_core_brain_kg_message_index
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_core_brain_kg_message_index PRIVATE cxx_std_17)
gtest_discover_tests(unit_core_brain_kg_message_index
    PROPERTIES LABELS "unit;core;brain_kg;phase6")
