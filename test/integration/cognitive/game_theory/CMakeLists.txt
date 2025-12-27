# ============================================================================
# Game Theory Integration Tests
# ============================================================================

# Game-Theoretic Global Workspace Integration Test
add_executable(integration_cognitive_gt_global_workspace
    test_gt_global_workspace_integration.cpp
)

target_link_libraries(integration_cognitive_gt_global_workspace
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_gt_global_workspace
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_cognitive_gt_global_workspace PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_gt_global_workspace
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;game_theory;global_workspace"
        TIMEOUT 120
)

# Game-Theoretic Hemispheric Integration Test
add_executable(integration_cognitive_gt_hemispheric
    test_gt_hemispheric_integration.cpp
)

target_link_libraries(integration_cognitive_gt_hemispheric
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(integration_cognitive_gt_hemispheric
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(integration_cognitive_gt_hemispheric PRIVATE cxx_std_17)

gtest_discover_tests(integration_cognitive_gt_hemispheric
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "integration;cognitive;game_theory;hemispheric"
        TIMEOUT 120
)
