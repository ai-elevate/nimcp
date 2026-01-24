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
