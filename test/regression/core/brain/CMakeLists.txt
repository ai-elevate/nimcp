# Brain Regression Tests
# Test core brain functionality regressions

# Note: factory and processing subdirectories are added separately by main CMakeLists.txt

# Oscillation Backward Compatibility Test
add_executable(regression_core_brain_oscillation_backward_compat
    ../test_oscillation_backward_compat.cpp
)
target_link_libraries(regression_core_brain_oscillation_backward_compat
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_oscillation_backward_compat
         COMMAND regression_core_brain_oscillation_backward_compat)

# Brain JSON Regression Test
add_executable(regression_core_brain_json_regression
    test_brain_json_regression.cpp
)
target_link_libraries(regression_core_brain_json_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_json_regression
         COMMAND regression_core_brain_json_regression)

# Distributed COW Regression Test
add_executable(regression_core_brain_distributed_cow_regression
    test_distributed_cow_regression.cpp
)
target_link_libraries(regression_core_brain_distributed_cow_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_distributed_cow_regression
         COMMAND regression_core_brain_distributed_cow_regression)

# Memory Tracking Regression Test
add_executable(regression_core_brain_memory_tracking_regression
    test_memory_tracking_regression.cpp
)
target_link_libraries(regression_core_brain_memory_tracking_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
    )
add_test(NAME regression_core_brain_memory_tracking_regression
         COMMAND regression_core_brain_memory_tracking_regression)

# Brain Persistence Regression Test - DISABLED: outdated API
# add_executable(regression_core_brain_persistence_regression
#     test_brain_persistence_regression.cpp
# )
# target_link_libraries(regression_core_brain_persistence_regression
#     PRIVATE
#         GTest::GTest
#         GTest::Main
#         nimcp
#     )
# add_test(NAME regression_core_brain_persistence_regression
#          COMMAND regression_core_brain_persistence_regression)

# Brain Bio-Async Regression Test (DISABLED - needs API fixes)
# add_executable(regression_core_brain_bio_async_regression
#     test_brain_bio_async_regression.cpp
# )
# target_link_libraries(regression_core_brain_bio_async_regression
#     PRIVATE
#         GTest::GTest
#         GTest::Main
#         nimcp
#     )
# add_test(NAME regression_core_brain_bio_async_regression
#          COMMAND regression_core_brain_bio_async_regression)

# Brain Coordinator Regression Test
add_executable(regression_core_brain_coordinator_regression
    test_brain_coordinator_regression.cpp
)
target_link_libraries(regression_core_brain_coordinator_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
    )
target_include_directories(regression_core_brain_coordinator_regression
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME regression_core_brain_coordinator_regression
         COMMAND regression_core_brain_coordinator_regression)
set_tests_properties(regression_core_brain_coordinator_regression PROPERTIES LABELS "regression")

# Set test timeouts for long-running regression tests (5 minutes)
# set_tests_properties(regression_core_brain_bio_async_regression
#     PROPERTIES TIMEOUT 300)
# set_tests_properties(regression_core_brain_persistence_regression
#     PROPERTIES TIMEOUT 300)

# KG Hierarchy Graph Algorithms Regression Test
add_executable(regression_core_brain_kg_hierarchy_graph_algorithms
    test_kg_hierarchy_graph_algorithms_regression.cpp
)
target_link_libraries(regression_core_brain_kg_hierarchy_graph_algorithms
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        Threads::Threads
    )
target_include_directories(regression_core_brain_kg_hierarchy_graph_algorithms
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME regression_core_brain_kg_hierarchy_graph_algorithms
         COMMAND regression_core_brain_kg_hierarchy_graph_algorithms)
set_tests_properties(regression_core_brain_kg_hierarchy_graph_algorithms PROPERTIES LABELS "regression")

# Add subcortical subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/subcortical")
    add_subdirectory(subcortical)
endif()

# Add regions subdirectory (hypothalamus, broca, cerebellum, etc.)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regions")
    add_subdirectory(regions)
endif()

# Add hemispheric subdirectory (hemispheric brain, corpus callosum, lateralization)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hemispheric")
    add_subdirectory(hemispheric)
endif()
