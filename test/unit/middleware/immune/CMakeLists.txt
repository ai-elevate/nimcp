# Unit tests for middleware immune integration

# Test: Pattern Immune Integration
add_executable(unit_middleware_immune_pattern_immune
    test_pattern_immune.cpp
)

target_link_libraries(unit_middleware_immune_pattern_immune
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_include_directories(unit_middleware_immune_pattern_immune PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_middleware_immune_pattern_immune
         COMMAND unit_middleware_immune_pattern_immune)

# Set test properties
set_tests_properties(unit_middleware_immune_pattern_immune PROPERTIES
    TIMEOUT 60
    LABELS "unit;middleware;immune;pattern"
)

# Test: Routing Immune Integration
add_executable(unit_middleware_immune_routing_immune_integration
    test_routing_immune_integration.cpp
)

target_link_libraries(unit_middleware_immune_routing_immune_integration
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_include_directories(unit_middleware_immune_routing_immune_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_middleware_immune_routing_immune_integration
         COMMAND unit_middleware_immune_routing_immune_integration)

# Set test properties
set_tests_properties(unit_middleware_immune_routing_immune_integration PROPERTIES
    TIMEOUT 30
    LABELS "unit;middleware;immune;routing"
)

# Test: Training Immune Integration
add_executable(unit_middleware_immune_training_immune
    test_training_immune.cpp
)

target_link_libraries(unit_middleware_immune_training_immune
    nimcp
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_include_directories(unit_middleware_immune_training_immune PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_middleware_immune_training_immune
         COMMAND unit_middleware_immune_training_immune)

# Set test properties
set_tests_properties(unit_middleware_immune_training_immune PROPERTIES
    TIMEOUT 60
    LABELS "unit;middleware;immune;training"
)
