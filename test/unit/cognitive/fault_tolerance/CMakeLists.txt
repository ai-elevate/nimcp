# Cognitive Fault Tolerance Unit Tests

# Emotional Tagging Unit Tests
add_executable(unit_cognitive_fault_tolerance_emotional_tagging
    test_emotional_tagging.cpp
)
target_link_libraries(unit_cognitive_fault_tolerance_emotional_tagging
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_fault_tolerance_emotional_tagging
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_cognitive_fault_tolerance_emotional_tagging
         COMMAND unit_cognitive_fault_tolerance_emotional_tagging)
set_tests_properties(unit_cognitive_fault_tolerance_emotional_tagging PROPERTIES LABELS "unit")

# Self-Repair Coordinator Unit Tests
add_executable(unit_cognitive_fault_tolerance_self_repair
    test_self_repair.cpp
)
target_link_libraries(unit_cognitive_fault_tolerance_self_repair
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_cognitive_fault_tolerance_self_repair
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
)
add_test(NAME unit_cognitive_fault_tolerance_self_repair
         COMMAND unit_cognitive_fault_tolerance_self_repair)
set_tests_properties(unit_cognitive_fault_tolerance_self_repair PROPERTIES LABELS "unit;self-repair")
