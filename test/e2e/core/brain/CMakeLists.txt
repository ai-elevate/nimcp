# ============================================================================
# NIMCP Core Brain Layer End-to-End Tests
# ============================================================================
# WHAT: Build configuration for core brain layer E2E tests
# WHY:  Test complete brain pipelines including hemispheric and thalamic layers
# HOW:  Include subdirectories for each brain module
# ============================================================================

message(STATUS "")
message(STATUS "-- Configuring Core Brain Layer E2E Tests")

# Include Hemispheric Brain E2E tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/hemispheric/CMakeLists.txt)
    add_subdirectory(hemispheric)
endif()

# Include Brain Regions E2E tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/regions/CMakeLists.txt)
    add_subdirectory(regions)
endif()

# Include Thalamic Layer E2E tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/thalamic/CMakeLists.txt)
    add_subdirectory(thalamic)
endif()

# Include Subcortical E2E tests (basal ganglia pipeline)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/subcortical/CMakeLists.txt)
    add_subdirectory(subcortical)
endif()

# Include Language Processing E2E tests (Broca/Wernicke)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/language/CMakeLists.txt)
    add_subdirectory(language)
endif()

# Brain Cycle Coordinator E2E Test (Full system health tracking)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_brain_cycle_coordinator_e2e.cpp)
    add_executable(e2e_brain_cycle_coordinator
        test_brain_cycle_coordinator_e2e.cpp
    )
    target_link_libraries(e2e_brain_cycle_coordinator
        PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
    )
    target_include_directories(e2e_brain_cycle_coordinator
        PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
    )
    gtest_discover_tests(e2e_brain_cycle_coordinator
        PROPERTIES LABELS "e2e;core;brain;cycle_coordinator")
endif()
